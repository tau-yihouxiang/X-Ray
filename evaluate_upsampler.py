import glob
import random
from diffusers.utils import load_image
import torch
from PIL import Image
import os
import numpy as np
import torchvision
import open3d as o3d
import torch.nn.functional as F
from tqdm import tqdm
from src.chamfer_distance import compute_trimesh_chamfer
from scipy.sparse import csr_matrix
import argparse
from diffusers import AutoencoderKL
from src.xray_decoder import AutoencoderKLTemporalDecoder



def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    
    # The origin of all rays is the camera origin in world coordinate
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape) # (H, W, 3)

    return rays_o, rays_d

def depth_to_pcd_normals(GenDepths, GenNormals, GenColors):
    camera_angle_x = 0.8575560450553894
    image_width = GenDepths.shape[-1]
    image_height = GenDepths.shape[-2]
    fx = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
            2, image_height * image_width).T  # [h, w, 2]

    grid = rays_screen_coords.reshape(image_height, image_width, 2)

    cx = image_width / 2.0
    cy = image_height / 2.0

    i, j = grid[..., 1], grid[..., 0]

    directions = np.stack([(i-cx)/fx, -(j-cy)/fx, -np.ones_like(i)], -1) # (H, W, 3)

    c2w = np.eye(4).astype(np.float32)

    rays_origins, ray_directions = get_rays(directions, c2w)
    rays_origins = rays_origins[None].repeat(GenDepths.shape[0], 0)
    ray_directions = ray_directions[None].repeat(GenDepths.shape[0], 0)

    GenDepths = GenDepths.transpose(0, 2, 3, 1)
    GenNormals = GenNormals.transpose(0, 2, 3, 1)
    GenColors = GenColors.transpose(0, 2, 3, 1)
    
    valid_index = GenDepths[..., 0] > 0
    rays_origins = rays_origins[valid_index]
    ray_directions = ray_directions[valid_index]
    GenDepths = GenDepths[valid_index]
    normals = GenNormals[valid_index]
    colors = GenColors[valid_index]
    xyz = rays_origins + ray_directions * GenDepths

    return xyz, normals, colors


def load_depths(depths_path):
    # 加载稀疏矩阵数据
    loaded_data = np.load(depths_path)
    # 从加载的数据中获取稀疏矩阵的组成部分
    loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    # 将稀疏矩阵还原为原始形状
    original_shape = (16, 1+3+3, 256, 256)
    restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
    return restored_array

if __name__ == "__main__":

    parser = argparse.ArgumentParser("SVD Depth Inference")
    parser.add_argument("--exp", type=str, default="Objaverse_80K_upsampler", help="experiment name")
    parser.add_argument("--data_root", type=str, default="/data/taohu/Data/Objaverse/Objaverse_80K_filter/depths", help="data root")
    args = parser.parse_args()

    near = 0.6
    far = 2.4
    num_frames = 8

    exp_name = args.exp
    xray_root = args.data_root

    vae_image = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).cuda()

    # Get the most recent checkpoint
    dirs = os.listdir(os.path.join("Output", exp_name))
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    ckpt_name = dirs[-1]
    print("restore from", f"Output/{exp_name}/{ckpt_name}/vae")

    vae = AutoencoderKLTemporalDecoder.from_pretrained(f"Output/{exp_name}/{ckpt_name}", subfolder="vae").cuda()

    height = 256
    width = 256

    xray_paths = glob.glob(os.path.join(xray_root, "*/*.npz"))
    image_paths = [x.replace("depths", "images").replace(".npz", ".png") for x in xray_paths]
    sorted(image_paths)
    image_paths = image_paths[::10] # test set
    image_paths = image_paths[::50]

    all_chamfer_distance = []
    progress_bar =  tqdm(range(len(image_paths)))
    for i in progress_bar:
        image_path = image_paths[i]
        uid = image_path.split("/")[-2]

        with torch.no_grad():
            depth_path = image_path.replace("images", "depths").replace(".png", ".npz")
            depths = load_depths(depth_path)
            xray = torch.from_numpy(depths).float()[:8]  # [8, 7, H, W]
            hit = (xray[:, 0:1] > 0).clone().float() * 2 - 1
            xray[:, 0] = (xray[:, 0] - near) / (far - near) * 2 - 1 # (0-0.6) / (2.4 - 0.6) = -0.3333
            xray[:, 1:4] = F.normalize(xray[:, 1:4], dim=1)
            xray[:, 4:7] = xray[:, 4:7] * 2 - 1
            xray = torch.cat([xray[:, 0:1], xray[:, 4:7], hit], dim=1)
            xray_low = torch.nn.functional.interpolate(xray, size=(height // 4, width // 4), mode="nearest")[None].cuda()

            xray_low = xray_low + torch.randn_like(xray_low) * random.uniform(0, 1) * 0.1

            image = load_image(image_path).resize((width * 2, height * 2), Image.BILINEAR).convert("RGB")
            conditional_pixel_values = (torchvision.transforms.ToTensor()(image).unsqueeze(0) * 2 - 1).half().cuda()
            conditional_latents = vae_image.encode(conditional_pixel_values).latent_dist.mode().float()

            # Concatenate the `conditional_latents` with the `noisy_latents`.
            conditional_latents = conditional_latents.unsqueeze(
                1).repeat(1, xray_low.shape[1], 1, 1, 1).float()
            xray_input = torch.cat(
                [xray_low, conditional_latents], dim=2)
            
            xray_input = xray_input.flatten(0, 1)
            model_pred = vae(xray_input, num_frames=num_frames).sample
            outputs = model_pred.reshape(-1, num_frames, *model_pred.shape[1:])[0]
            outputs = outputs.clamp(-1, 1) # clamp to [-1, 1]

        os.makedirs(f"Output/{exp_name}/evaluate", exist_ok=True)
        img = Image.open(image_path).resize((width * 2, height * 2), Image.BILINEAR)
        # img_write = Image.new("RGB", img.size, (255, 255, 255))
        # img_write.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        img.save(f"Output/{exp_name}/evaluate/{uid}_original.png")

        GenDepths = (outputs[:, 0:1] * 0.5 + 0.5) * (far - near) + near
        GenHits = (outputs[:, 7:8] > 0).float()
        GenDepths[GenHits == 0] = 0
        GenDepths[GenDepths <= near] = 0
        GenDepths[GenDepths >= far] = 0
        GenNormals = F.normalize(outputs[:, 1:4], dim=1)
        GenNormals[GenHits.repeat(1, 3, 1, 1) == 0] = 0
        GenColors = outputs[:, 4:7] * 0.5 + 0.5
        GenColors[GenHits.repeat(1, 3, 1, 1) == 0] = 0

        GenSurfaces = (GenDepths > 0).float().repeat(1, 3, 1, 1)
        visual_image = torch.stack([GenSurfaces, GenDepths.repeat(1, 3, 1, 1), GenNormals, GenColors], dim=1)
        visual_image = visual_image.reshape(-1, 3, 256, 256)
        torchvision.utils.save_image(visual_image, f"Output/{exp_name}/evaluate/{uid}_xray.png", nrow=4, pad_value=255)

        GenDepths = GenDepths.cpu().numpy()
        GenNormals = GenNormals.cpu().numpy()
        GenColors = GenColors.cpu().numpy()

        gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths, GenNormals, GenColors)
        # gen_pts[:, 2] += 1.5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gen_pts)
        pcd.normals = o3d.utility.Vector3dVector(gen_normals)
        pcd.colors = o3d.utility.Vector3dVector(gen_colors)
        o3d.io.write_point_cloud(f"Output/{exp_name}/evaluate/{uid}_prd.ply", pcd)
        
        gt_path = image_path.replace("images", "depths").replace(".png", ".npz")
        xray = load_depths(gt_path)[:num_frames]
        xray = torch.from_numpy(xray)
        xray = xray.cpu().numpy()
        GtDepths = xray[:, 0:1]
        GtNormals = xray[:, 1:4]
        GtColors = xray[:, 4:7]
        gt_pts, gt_normals, gt_colors = depth_to_pcd_normals(GtDepths, GtNormals, GtColors)
        # gt_pts[:, 2] += 1.5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_pts)
        pcd.normals = o3d.utility.Vector3dVector(gt_normals)
        pcd.colors = o3d.utility.Vector3dVector(gt_colors)
        o3d.io.write_point_cloud(f"Output/{exp_name}/evaluate/{uid}_gt.ply", pcd)

        chamfer_distance = compute_trimesh_chamfer(gt_pts, gen_pts)
        all_chamfer_distance += [chamfer_distance]
        progress_bar.set_postfix({"chamfer_distance": np.mean(all_chamfer_distance)})
        progress_bar.update(1)

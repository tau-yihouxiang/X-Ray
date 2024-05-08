import glob
import time
from diffusers import UNetSpatioTemporalConditionModel
from diffusers.utils import load_image
import torch
from PIL import Image
import os
import numpy as np
import trimesh
import torchvision
import open3d as o3d
import torch.nn.functional as F
import shutil
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
    loaded_data = np.load(depths_path)
    loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    original_shape = (16, 1+3+3, 256, 256)
    restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
    return restored_array

if __name__ == "__main__":

    parser = argparse.ArgumentParser("SVD Depth Inference")
    parser.add_argument("--exp", type=str, default="Objaverse_80K_up_5to8", help="experiment name")
    parser.add_argument("--data_root", type=str, default="Output/train_with_A100/evaluate", help="data root")
    args = parser.parse_args()

    near = 0.6
    far = 2.4
    num_frames = 8

    exp_name = args.exp
    xray_root = args.data_root

    vae_image = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).cuda()

    xray_paths = glob.glob(os.path.join(xray_root, "*/*.npz"))
    image_paths = [x.replace("depths", "images").replace(".npz", ".png") for x in xray_paths]
    sorted(image_paths)
    
    os.makedirs(f"Output/{exp_name}/evaluate", exist_ok=True)
    progress_bar =  tqdm(range(len(image_paths)))
    
    # Get the most recent checkpoint
    dirs = os.listdir(os.path.join("Output", exp_name))
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    ckpt_name = dirs[-1]
    print("restore from", f"Output/{exp_name}/{ckpt_name}/vae")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(f"Output/{exp_name}/{ckpt_name}", subfolder="vae").cuda()

    height = 256
    width = 256

    all_chamfer_distance = []
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        uid = os.path.basename(image_path).replace(".png", "")

        with torch.no_grad():
            depth_path = image_path.replace(".png", ".pt")
            depths = torch.load(depth_path)
            xray_low = depths.clone()  # [8, 7, H, W]
            hit = (xray_low[:, 0:1] > -1).clone().float() * 2 - 1
            # xray_low = torch.cat([xray_low, hit], dim=1).cuda()[None]
            xray_low = torch.cat([xray_low[:, 0:1], xray_low[:, 4:7], hit], dim=1).cuda()[None]

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

        image.save(f"Output/{exp_name}/evaluate/{uid}.png")
        GenDepths = (outputs[:, 0:1] * 0.5 + 0.5) * (far - near) + near
        GenHits = (outputs[:, 7:8] > 0).float()
        GenDepths[GenHits == 0] = 0
        GenDepths[GenDepths <= near] = 0
        GenDepths[GenDepths >= far] = 0
        GenNormals = F.normalize(outputs[:, 1:4], dim=1)
        GenNormals[GenHits.repeat(1, 3, 1, 1) == 0] = 0
        GenColors = outputs[:, 4:7] * 0.5 + 0.5
        GenColors[GenHits.repeat(1, 3, 1, 1) == 0] = 0

        GenDepths = GenDepths.cpu().numpy()
        GenNormals = GenNormals.cpu().numpy()
        GenColors = GenColors.cpu().numpy()

        gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths, GenNormals, GenColors)
        # gen_pts += 1.5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gen_pts)
        pcd.normals = o3d.utility.Vector3dVector(gen_normals)
        pcd.colors = o3d.utility.Vector3dVector(gen_colors)
        o3d.io.write_point_cloud(f"Output/{exp_name}/evaluate/{uid}_prd_up4x.ply", pcd)

        # copy prd ply to the folder
        shutil.copy(image_path.replace(".png", "_prd.ply"), f"Output/{exp_name}/evaluate/{uid}_prd.ply")
        gt_pcd = o3d.io.read_point_cloud(image_path.replace(".png", "_gt.ply"))
        gt_pts = np.asarray(gt_pcd.points)
        # gt_pts += 1.5

        chamfer_distance = compute_trimesh_chamfer(gt_pts, gen_pts)
        all_chamfer_distance += [chamfer_distance]
        progress_bar.set_postfix({"chamfer_distance": np.mean(all_chamfer_distance)})
        progress_bar.update(1)
    print(f"{ckpt_name}: chamfer distance: {np.mean(all_chamfer_distance)}")

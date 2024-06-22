import glob
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
from src.metrics import chamfer_distance_and_f_score
from scipy.sparse import csr_matrix
import argparse
from diffusers import AutoencoderKLTemporalDecoder
from src.dataset import UpsamplerDataset
import time
from torch.utils.tensorboard import SummaryWriter


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    
    # The origin of all rays is the camera origin in world coordinate
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape) # (H, W, 3)

    return rays_o, rays_d

def xray_to_pcd(GenDepths, GenNormals, GenColors):
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


def load_xray(xray_path):
    loaded_data = np.load(xray_path)
    loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    original_shape = (16, 1+3+3, 256, 256)
    restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
    return restored_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser("X-Ray full Inference")
    parser.add_argument("--exp_vae", type=str, help="experiment name")
    parser.add_argument("--data_root", type=str, default="Data/ShapeNetV2_Car", help="data root")
    args = parser.parse_args()

    near = 0.6
    far = 1.8
    
    num_frames = 8
    height = 256
    width = 256

    exp_vae = args.exp_vae
    writer = SummaryWriter(f"Output/{exp_vae}/metrics")

    while True:
        if os.path.exists(f"Output/{exp_vae}/evaluate"):
            shutil.rmtree(f"Output/{exp_vae}/evaluate")

        os.makedirs(f"Output/{exp_vae}/evaluate", exist_ok=True)
        progress_bar =  tqdm(range(500))
        
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join("Output", exp_vae))
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        ckpt_name = dirs[-1]
        print("restore from", f"Output/{exp_vae}/{ckpt_name}/vae")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(f"Output/{exp_vae}/{ckpt_name}", subfolder="vae").cuda()

        all_chamfer_distance = []
        all_f_score = []

        dataset = UpsamplerDataset(args.data_root, height, num_frames, near=near, far=far, phase="val")

        for i in progress_bar:
            image_path = dataset[i]["image_path"]
            uid = image_path.split("/")[-2]

            xray = dataset[i]["xray"].cuda()[None]
            xray_input = xray.flatten(0, 1)
            with torch.no_grad():
                model_pred = vae(xray_input, num_frames=num_frames).sample
            outputs = model_pred.reshape(-1, num_frames, *model_pred.shape[1:])[0]
            outputs = outputs.detach().clip(-1, 1)

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

            GenDepths_ori = GenDepths.copy()
            for i in range(GenDepths.shape[0]-1):
                GenDepths[i+1] = np.where(GenDepths_ori[i+1] < GenDepths_ori[i], 0, GenDepths_ori[i+1])

            gen_pts, gen_normals, gen_colors = xray_to_pcd(GenDepths, GenNormals, GenColors)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gen_pts)
            pcd.normals = o3d.utility.Vector3dVector(gen_normals)
            pcd.colors = o3d.utility.Vector3dVector(gen_colors)
            o3d.io.write_point_cloud(f"Output/{exp_vae}/evaluate/{uid}_prd.ply", pcd)

            gt_path = image_path.replace("images", "xrays").replace(".png", ".npz")
            xray_gt = load_xray(gt_path)[:8]
            GtDepths = xray_gt[:, 0:1]
            GtNormals = xray_gt[:, 1:4]
            GtColors = xray_gt[:, 4:7]
            gt_pts, gt_normals, gt_colors = xray_to_pcd(GtDepths, GtNormals, GtColors)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.normals = o3d.utility.Vector3dVector(gt_normals)
            pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)
            o3d.io.write_point_cloud(f"Output/{exp_vae}/evaluate/{uid}_gt.ply", pcd_gt)

            chamfer_distance, f_score = chamfer_distance_and_f_score(gt_pts, gen_pts, threshold=0.01)
            all_chamfer_distance += [chamfer_distance]
            all_f_score += [f_score]

            progress_bar.set_postfix({"CD": np.mean(all_chamfer_distance),
                                    "FS@0.01": np.mean(all_f_score)})
            progress_bar.update(1)

        print(f"{ckpt_name}: CD: {np.mean(all_chamfer_distance)}")
        print(f"{ckpt_name}: FS@0.01: {np.mean(all_f_score)}")
        
        # Write the final CD and FS@0.01 to TensorBoard
        global_step = ckpt_name.split("-")[1]
        writer.add_scalar("Chamfer Distance", np.mean(all_chamfer_distance), global_step=global_step)
        writer.add_scalar("F-Score @ 0.01", np.mean(all_f_score), global_step=global_step)

        # sleep for 30 minutes
        time.sleep(60 * 30)
        print("Sleeping for 30 minutes")

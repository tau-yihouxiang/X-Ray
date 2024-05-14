import glob
import os
import shutil
import imageio
import numpy as np
import tqdm
import PIL
from PIL import Image
import time
import random
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
import open3d as o3d
import torchvision

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


def load_depths( depths_path):
	loaded_data = np.load(depths_path)

	loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])

	original_shape = (16, 7, 256, 256)
	restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
	return restored_array

instance_data_root = "Data/Objaverse_XRay/depths/a57dd10038a14ef8a141e0e7c3bc3e27"
# mesh_dir = "/data/taohu/Data/ShapeNet/ShapeNetCore.v2/02958343"

depths_paths = glob.glob(os.path.join(instance_data_root, "**/*.npz"), recursive=True)
# shuffle
random.shuffle(depths_paths)

near = 0.6
far = 1.8

for depth_path in depths_paths:
    print(depth_path)

    # depth_path = "Data/Objaverse_XRay/depths/a57dd10038a14ef8a141e0e7c3bc3e27/000.npz"

    depths = load_depths(depth_path)
    depths = torch.from_numpy(depths).float()
    GenDepths = depths[:, 0:1].expand(-1, 3, -1, -1)
    GenDepths = ((GenDepths - near) / (far - near)).clip(min=0, max=1)
    GenNormals = depths[:, 1:4]
    GenNormals = F.normalize(GenNormals, p=2, dim=1)
    GenNormals = (GenNormals * 0.5 + 0.5).clip(min=0, max=1)
    GenColors = depths[:, 4:7].clip(min=0, max=1)
    GenHits = (GenDepths > 0).float()

    # gray
    GenDepths[GenHits == 0] = 1
    GenNormals[GenHits == 0] = 1
    GenColors[GenHits == 0] = 1

    GenHits = 1 - GenHits
    GenHits[GenHits == 0] = 0.5

    XRay = torch.stack([GenHits, GenDepths, GenNormals, GenColors], dim=0)
    XRay = XRay.reshape(4 * 16, 3, 256, 256)
    torchvision.utils.save_image(XRay, "logs/xray.png", nrow=16, padding=0)

    # save image
    image_path = depth_path.replace("depths", "images").replace("npz", "png")
    image = Image.open(image_path)
    # convert to rgba image to white color image using PIL
    white_image = Image.new("RGB", image.size, "WHITE")
    white_image.paste(image, (0, 0), image)
    white_image.save("logs/image.png")

    # # copy mesh file to logs
    # mesh_path = os.path.join(mesh_dir, depth_path.split("/")[-2], "models")
    # # copy mesh path to logs
    # shutil.copytree(mesh_path, "logs/mesh")
    import pdb; pdb.set_trace()

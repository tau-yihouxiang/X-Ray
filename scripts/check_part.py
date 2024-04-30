import glob
import os
import imageio
import numpy as np
from PIL import Image
import random
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
import open3d as o3d
import torchvision
import shutil

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

instance_data_root = "Data/ShapeNet_Car/depths"

depths_paths = glob.glob(os.path.join(instance_data_root, "*/*/*.npz"))
# shuffle
random.shuffle(depths_paths)

near = 0.6
far = 2.4

for depth_path in depths_paths:
    print(depth_path)
    depths = load_depths(depth_path)
    GenDepths = depths[:, 0:1]
    GenNormals = depths[:, 1:4]
    GenNormals = GenNormals / (np.linalg.norm(GenNormals, axis=1, keepdims=True) + 1e-8)
    GenColors = depths[:, 4:7]

    torchvision.utils.save_image(torch.tensor((GenDepths - near) / (far - near)), "Output/depths.png", nrow=4)
    torchvision.utils.save_image(torch.tensor(GenNormals * 0.5 + 0.5), "Output/normals.png", nrow=4)
    torchvision.utils.save_image(torch.tensor(GenColors), "Output/colors.png", nrow=4)

    # save image
    image_path = depth_path.replace("depths", "images").replace("npz", "png")
    image = Image.open(image_path)
    image.save("Output/image.png")

    gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths, GenNormals, GenColors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gen_pts)
    pcd.normals = o3d.utility.Vector3dVector(gen_normals)
    pcd.colors = o3d.utility.Vector3dVector(gen_colors)
    o3d.io.write_point_cloud("Output/gt.ply", pcd)
    
    # remove path "Output/parts"
    shutil.rmtree("Output/parts", ignore_errors=True)
    os.makedirs("Output/parts", exist_ok=True)
    for i in range(16):
        gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths[i:i+1], GenNormals[i:i+1], GenColors[i:i+1])
        if len(gen_pts) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gen_pts)
        pcd.normals = o3d.utility.Vector3dVector(gen_normals)
        pcd.colors = o3d.utility.Vector3dVector(gen_colors)
        o3d.io.write_point_cloud(f"Output/parts/part_{i:02d}.ply", pcd)

    # save GenDepths, GenNormals, GenColors as a sequential video
    GenDepths = GenDepths / (far - near)
    GenHits = (GenDepths > 0).astype(np.float32)
    GenNormals = GenNormals * 0.5 + 0.5
    GenColors = GenColors

    GenHits = GenHits * 255
    GenDepths = GenDepths * 255
    GenNormals = GenNormals * 255
    GenColors = GenColors * 255

    GenDepths = GenDepths.astype(np.uint8).transpose(0, 2, 3, 1).repeat(3, -1)
    GenHits = GenHits.astype(np.uint8).transpose(0, 2, 3, 1).repeat(3, -1)
    GenNormals = GenNormals.astype(np.uint8).transpose(0, 2, 3, 1)
    GenColors = GenColors.astype(np.uint8).transpose(0, 2, 3, 1)

    GenDepths[GenHits == 0] = 128
    GenNormals[GenHits == 0] = 128
    GenColors[GenHits == 0] = 128
    GenHits[GenHits == 0] = 128 

    # convert image background to white
    image = image.resize((256, 256))
    white_image = Image.new("RGB", image.size, (128, 128, 128))
    # paste the image on the white background
    white_image.paste(image, mask=image.split()[3])  
    white_image = np.array(white_image)
    # repeat the image to match the number of frames
    white_image = white_image[None].repeat(GenHits.shape[0], 0)

    GenXRay = np.concatenate([white_image, GenHits, GenDepths, GenNormals, GenColors], axis=2)
    imageio.mimsave('Output/xray.gif', GenXRay, loop=1024, format='GIF', fps=1)  # 'duration' controls the frame timing in seconds
    import pdb; pdb.set_trace()

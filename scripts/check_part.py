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
import trimesh

def create_arrow(start, end, shaft_diameter=0.002, head_diameter=0.01, head_length=0.02):
    # Vector from start to end
    direction = np.array(end) - np.array(start)
    arrow_length = np.linalg.norm(direction)
    direction = direction / arrow_length  # Normalize the direction vector

    # Create the shaft of the arrow
    shaft_length = arrow_length 
    shaft = trimesh.creation.cylinder(radius=shaft_diameter / 2, height=shaft_length,
                                      sections=32, transform=None)

    # Create the head of the arrow
    head = trimesh.creation.cone(radius=head_diameter / 2, height=head_length,
                                 sections=32, transform=None)

    # Position the shaft
    shaft_transform = trimesh.transformations.translation_matrix(start + direction * shaft_length / 2)
    shaft_transform = np.dot(shaft_transform, trimesh.transformations.rotation_matrix(
        angle=np.arccos(np.dot([0, 0, 1], direction)),
        direction=np.cross([0, 0, 1], direction),
        point=[0, 0, 0]))
    shaft.apply_transform(shaft_transform)

    # Position the head
    head_transform = trimesh.transformations.translation_matrix(start + direction * (arrow_length - head_length / 2))
    head_transform = np.dot(head_transform, trimesh.transformations.rotation_matrix(
        angle=np.arccos(np.dot([0, 0, 1], direction)),
        direction=np.cross([0, 0, 1], direction),
        point=[0, 0, 0]))
    head.apply_transform(head_transform)

    # Combine shaft and head to form the arrow
    arrow_mesh = shaft + head

    # Set the color of the arrow to #d1923f
    arrow_mesh.visual.face_colors = [209, 146, 63, 64]
    return arrow_mesh


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

	original_shape = (16, 7, 1024, 1024)
	restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
	return restored_array

instance_data_root = "Data/ShapeNet_Car/depths_1024"

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

    # vis
    nohit = GenDepths == 0
    D = GenDepths.copy()
    D[nohit] = (far + near) / 2
    N = GenNormals.copy()
    N[nohit.repeat(3, 1)] = 0
    C = GenColors.copy()
    C[nohit.repeat(3, 1)] = 0.5

    torchvision.utils.save_image(torch.tensor((D - near) / (far - near)), "Output/depths.png", nrow=16, padding=0)
    torchvision.utils.save_image(torch.tensor(N * 0.5 + 0.5), "Output/normals.png", nrow=16, padding=0)
    torchvision.utils.save_image(torch.tensor(C), "Output/colors.png", nrow=16, padding=0)

    # save image
    image_path = depth_path.replace("depths_1024", "images").replace("npz", "png")
    image = Image.open(image_path)
    image.save("Output/image.png")

    gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths, GenNormals, GenColors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gen_pts)
    pcd.normals = o3d.utility.Vector3dVector(gen_normals)
    pcd.colors = o3d.utility.Vector3dVector(gen_colors)
    o3d.io.write_point_cloud("Output/gt.ply", pcd)

    shutil.rmtree("Output/parts")
    os.makedirs("Output/parts", exist_ok=True)
    
    for i in range(16):
        gen_pts, gen_normals, gen_colors = depth_to_pcd_normals(GenDepths[i:i+1], GenNormals[i:i+1], GenColors[i:i+1])
        if len(gen_pts) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gen_pts)
        pcd.normals = o3d.utility.Vector3dVector(gen_normals)
        pcd.colors = o3d.utility.Vector3dVector(gen_colors)
        # export the point cloud to a ply file
        o3d.io.write_point_cloud(f"Output/parts/{i:02d}_object.ply", pcd)

        # create arrow that point to the pcd.points
        arrow_trimesh = []
        # random choise 100 points to create the arrow
        if len(gen_pts) > 100:
            idx = np.random.choice(len(gen_pts), 100, replace=False)
        else:
            idx = np.arange(len(gen_pts))
        for j in idx:
            arrow_trimesh += [create_arrow([0, 0, 0], gen_pts[j])]
        arrow_trimesh = trimesh.util.concatenate(arrow_trimesh)
        # export the arrow to a ply file
        arrow_trimesh.export(f"Output/parts/{i:02d}_arrow.ply")

    # merged_pcd.colors = o3d.utility.Vector3dVector(np.asarray(merged_pcd.normals) * 0.5 + 0.5)
    # o3d.io.write_point_cloud("Output/merged_normal.ply", merged_pcd)

    # merged_pcd.colors = o3d.utility.Vector3dVector(np.asarray(merged_pcd.colors) * 0.0 + 1.0)
    # o3d.io.write_point_cloud("Output/merged_depth.ply", merged_pcd)

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
    image = image.resize((1024, 1024))
    white_image = Image.new("RGB", image.size, (128, 128, 128))
    # paste the image on the white background
    white_image.paste(image, mask=image.split()[3])  
    white_image = np.array(white_image)
    # repeat the image to match the number of frames
    white_image = white_image[None].repeat(GenHits.shape[0], 0)

    GenXRay = np.concatenate([white_image, GenHits, GenDepths, GenNormals, GenColors], axis=2)
    imageio.mimsave('Output/xray.gif', GenXRay, loop=1024, format='GIF', fps=1)  # 'duration' controls the frame timing in seconds
    import pdb; pdb.set_trace()

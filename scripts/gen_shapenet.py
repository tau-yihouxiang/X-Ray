import glob
import shutil
import PIL
import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector
import trimesh
import random
from tqdm import tqdm
import cv2
import os
from collections import defaultdict
import time
import json
import imageio
from scipy.sparse import csr_matrix
import torchvision
import torch
import torch.nn.functional as F
import open3d as o3d

def load_depths(depths_path):
    # 加载稀疏矩阵数据
    loaded_data = np.load(depths_path)

    # 从加载的数据中获取稀疏矩阵的组成部分
    loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])

    # 将稀疏矩阵还原为原始形状
    original_shape = (16, image_height, image_width)
    restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
    return restored_array


def load_from_json(fname):
    with open(fname, "r") as f:
        return json.load(f)

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape) # (H, W, 3)

    return rays_o, rays_d

class RaycastingImaging:
    def __init__(self):
        self.rays_screen_coords, self.rays_origins, self.rays_directions = None, None, None

    def __del__(self):
        del self.rays_screen_coords
        del self.rays_origins
        del self.rays_directions

    def prepare(self, image_height, image_width, intrinsics=None, c2w=None):
        # scanning radius is determined from the mesh extent
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays((image_height, image_width), intrinsics, c2w)

    def get_image(self, mesh):  #, features):
        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(mesh, self.rays_origins, self.rays_directions)

        # extract normals
        normals = mesh.face_normals[mesh_face_indexes]
        colors = mesh.visual.face_colors[mesh_face_indexes]

        mesh_face_indexes = np.unique(mesh_face_indexes)
        mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes])
        direction = self.rays_directions[ray_indexes][0]
        return ray_indexes, points, normals, colors, direction, mesh_vertex_indexes, mesh_face_indexes

        # assemble mesh fragment into a submesh
        # nbhood = reindex_zerobased(mesh, mesh_vertex_indexes, mesh_face_indexes)
        # return ray_indexes, points, normals, nbhood, mesh_vertex_indexes, mesh_face_indexes


def generate_rays(image_resolution, intrinsics, c2w):
    if isinstance(image_resolution, tuple):
        assert len(image_resolution) == 2
    else:
        image_resolution = (image_resolution, image_resolution)
    image_width, image_height = image_resolution

    # generate an array of screen coordinates for the rays
    # (rays are placed at locations [i, j] in the image)
    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
        2, image_height * image_width).T  # [h, w, 2]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    grid = rays_screen_coords.reshape(image_height, image_width, 2)
    
    i, j = grid[..., 1], grid[..., 0]
    directions = np.stack([(i-cx)/fx, -(j-cy)/fy, -np.ones_like(i)], -1) # (H, W, 3)

    rays_origins, ray_directions = get_rays(directions, c2w)
    rays_origins = rays_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    
    return rays_screen_coords, rays_origins, ray_directions


def ray_cast_mesh(mesh, rays_origins, ray_directions):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
        return_locations=True)
    return index_triangles, index_ray, point_cloud


from multiprocessing import Pool

def process_model(model_path):
    try:
        obj_id = model_path.split("/")[-3]
        json_path = os.path.join(img_dir, obj_id, "transforms.json")

        if not os.path.exists(json_path):
            print("skip")
            return
        
        meta = load_from_json(json_path)
        meta["frames"] = meta["frames"]
        
        if os.path.exists(os.path.join(depth_dir, obj_id, meta["frames"][-1]["file_path"][:-4] + ".npz")):
            print("existed")
            return
        
        mesh = trimesh.load(model_path,  force='mesh', process=False)

        if mesh.visual.kind != "texture":
            return

        # exchange axis
        mesh.vertices = mesh.vertices[:, [2, 0, 1]]
        # rotate -90 degree around z axis 
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 0, 1]))

        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)

        box_min, box_max = mesh.bounds
        # scale = np.max(np.abs(box_max - box_min))
        # mesh.apply_scale(1.1)

        box_min, box_max = mesh.bounds
        center = 0.0 * (box_min + box_max) / 2
        mesh.apply_translation(-center)

        mesh.visual = mesh.visual.to_color()
        if len(mesh.visual.vertex_colors.shape) == 1:
            mesh.visual.vertex_colors = np.tile(mesh.visual.vertex_colors[None], (len(mesh.vertices), 1))

        raycast = RaycastingImaging()
        
        for frame in (meta["frames"]):
            os.makedirs(os.path.join(depth_dir, obj_id), exist_ok=True)
            # save image
            c2w = np.array(frame["c2w"])

            Rt = np.linalg.inv(c2w)
            mesh_frame = mesh.copy().apply_transform(Rt)

            # # export mesh as ply using trimesh
            # mesh_frame.export(os.path.join(depth_dir, obj_id, frame["file_path"][:-4] + ".ply"))
            # # copy image
            # shutil.copy(os.path.join(img_dir, obj_id, frame["file_path"]), os.path.join(depth_dir, obj_id))
            # import pdb; pdb.set_trace()
            # break

            c2w = np.eye(4).astype(np.float32)[:3]

            camera_angle_x = float(meta["camera_angle_x"])
            focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

            cx = image_width / 2.0
            cy = image_height / 2.0

            intrinsics = np.array([[focal_length, 0, cx],
                                [0, focal_length, cy],
                                [0, 0, 1]])
            
            raycast.prepare(image_height=image_height, image_width=image_width, intrinsics=intrinsics, c2w=c2w)
            
            ray_indexes, points, normals, colors, direction, mesh_vertex_indexes, mesh_face_indexes = raycast.get_image(mesh_frame)   
            
            # normalize normals
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            colors = colors[:, :3] / 255.0

            # collect points and normals for each ray
            ray_points = defaultdict(list)
            ray_normals = defaultdict(list)
            ray_colors = defaultdict(list)
            for ray_index, point, normal, color in zip(ray_indexes, points, normals, colors):
                ray_points[ray_index].append(point)
                ray_normals[ray_index].append(normal)
                ray_colors[ray_index].append(color)

            # ray to image
            max_hits = 16
            GenDepths = np.zeros((max_hits, 1+3+3, 256, 256), dtype=np.float32)

            for i in range(max_hits):
                for ray_index, ray_point in ray_points.items():
                    if i < len(ray_point):
                        u = ray_index // 256
                        v = ray_index % 256
                        GenDepths[i, 0, u, v] = np.linalg.norm(ray_point[i] - c2w[:, 3])
                        GenDepths[i, 1:4, u, v] = ray_normals[ray_index][i]
                        GenDepths[i, 4:7, u, v] = ray_colors[ray_index][i]
            
            GenDepths = GenDepths.astype(np.float16)
            # save GenDepths as a npy file
            sparse_matrix = csr_matrix(GenDepths.reshape(16, -1))
            
            np.savez_compressed(os.path.join(depth_dir, obj_id, frame["file_path"][:-4]), 
                                data=sparse_matrix.data, 
                                indices=sparse_matrix.indices, 
                                indptr=sparse_matrix.indptr, 
                                shape=sparse_matrix.shape)
            # # export mesh
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.normals = o3d.utility.Vector3dVector(normals)
            # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            # o3d.io.write_point_cloud(os.path.join(depth_dir, obj_id, frame["file_path"][:-4] + ".ply"), pcd)

        print("saved")
    except Exception as e:
        print(e)
        return


if __name__ == "__main__":
    root_dir = "/data/taohu/Data/ShapeNet/ShapeNetCore.v2_Clean"
    img_dir = "/data/taohu/Data/ShapeNet/ShapeNetV2_Car/images"
    depth_dir = "/data/taohu/Data/ShapeNet/ShapeNetV2_Car/depths"
    image_height = 256
    image_width = 256

    model_paths = glob.glob(os.path.join(root_dir, "**/*.obj"), recursive=True)
    random.shuffle(model_paths)
    debug = False
    if debug:
        for model_path in tqdm(model_paths):
            process_model(model_path)
    else:
        with Pool(2) as p:
            print(p.map(process_model, model_paths))

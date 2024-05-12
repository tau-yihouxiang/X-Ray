import open3d as o3d
import numpy as np
import trimesh
import argparse
import os
import glob
from src.chamfer_distance import compute_trimesh_chamfer
import tqdm

def align_point_clouds(source, target):
    # scale aligned source point cloud
    pts_src = np.array(source.points)
    pts_src = pts_src - np.mean(pts_src, axis=0)
    scale_src = np.abs(pts_src).mean()

    pts_tgt = np.array(target.points)
    pts_tgt = pts_tgt - np.mean(pts_tgt, axis=0)
    scale_tgt = np.abs(pts_tgt).mean()
    source = source.scale(scale_tgt / scale_src, center=source.get_center())

    # Create a copy of the source point cloud
    source_copy = o3d.geometry.PointCloud(source)

    threshold = 0.01
    trans_init = np.identity(4).astype(np.float32)

    reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    # Apply the transformation to the source point cloud
    aligned_source = source_copy.transform(reg_p2p.transformation)

    return aligned_source


if __name__ == "__main__":

    parser = argparse.ArgumentParser("evaluate icp-cd")
    parser.add_argument("--out_dir", type=str, default="Output/Objaverse_XRay_upsampler/evaluate", help="output directory")
    args = parser.parse_args()

    # image files
    image_files = sorted(glob.glob(f"{args.out_dir}/*.png"))

    progress_bar = tqdm.tqdm(image_files)

    chamfer_distances = []
    for image_file in progress_bar:
        try:
            # Load the source and target point clouds
            uid = os.path.basename(image_file).replace(".png", "")
            src_path = f"{args.out_dir}/{uid}_prd.ply"
            if os.path.exists(src_path):
                # read point cloud
                source = o3d.io.read_point_cloud(src_path)
            else: # read mesh file them convert to point cloud using o3d
                source_mesh = trimesh.load_mesh(f"{args.out_dir}/{uid}_prd.obj")
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(source_mesh.vertices)

            target_path = f"{args.out_dir}/{uid}_gt.ply"
            target = o3d.io.read_point_cloud(target_path)

            # Align the point clouds
            aligned_source = align_point_clouds(source, target)

            # Save the aligned source point cloud
            aligned_source_path = f"{args.out_dir}/{uid}_prd_aligned.ply"
            o3d.io.write_point_cloud(aligned_source_path, aligned_source)

            # Compute the Chamfer distance
            chamfer_distance = compute_trimesh_chamfer(np.array(target.points), np.array(aligned_source.points))
            chamfer_distances.append(chamfer_distance)
            progress_bar.set_postfix({"chamfer_distance": np.mean(chamfer_distances)})
            progress_bar.update(1)
        except Exception as e:
            print(f"Error: {e}")

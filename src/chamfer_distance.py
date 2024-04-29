#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree


def compute_trimesh_chamfer(gt_points, gen_points):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer
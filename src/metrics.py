import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance_and_f_score(P, Q, threshold=0.1):
    kdtree_P = cKDTree(P)
    kdtree_Q = cKDTree(Q)
    
    dist_P_to_Q, _ = kdtree_P.query(Q)
    dist_Q_to_P, _ = kdtree_Q.query(P)
    
    chamfer_dist = np.mean(dist_P_to_Q) + np.mean(dist_Q_to_P)
    
    precision = np.mean(dist_P_to_Q < threshold)
    recall = np.mean(dist_Q_to_P < threshold)
    
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0.0
    
    return chamfer_dist, f_score

if __name__ == '__main__':
    P = np.random.rand(100, 3)  # Point cloud P with 100 points
    Q = np.random.rand(80, 3)   # Point cloud Q with 80 points

    chamfer_dist, f_score = chamfer_distance_and_f_score(P, Q, threshold=0.01)
    print(chamfer_dist, f_score, precision, recall)
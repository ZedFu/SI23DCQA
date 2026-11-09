import open3d as o3d
import numpy as np
import os
import time
import random
import argparse
from pathlib import Path
from data_utils import read_annotations, feature_dir
def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m == 0 or not np.isfinite(m):
        raise ValueError("Point cloud has zero extent or non-finite coordinates")
    pc = pc / m
    return pc

def farthest_point_sample(point, patch_size):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    # npoint = int(N/patch_size) + 1
    npoint = 6
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def knn_patch(pcd_name, patch_size = 2048):
    pcd = o3d.io.read_point_cloud(pcd_name)
    # print(np.array(pcd.points).shape)
    # print(np.array(pcd.colors).shape)
    # nomalize pc and set up kdtree
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {pcd_name}")
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    fps_point = farthest_point_sample(points,patch_size)
    
    
    point_size = fps_point.shape[0]


    patch_list = []

    for i in  range(point_size):
        [_,idx,dis] = kdtree.search_knn_vector_3d(fps_point[i],patch_size)
        #print(pc_normalize(np.asarray(point)[idx[1:], :]))
        patch_points = np.asarray(points)[idx[:], :]
        if patch_points.shape[0] < patch_size:
            patch_points = np.pad(patch_points,((0,patch_size - patch_points.shape[0]),(0,0)),mode='constant',constant_values=0)
        patch_list.append(patch_points) 
    
    #visualize(all_point(np.array(patch_list)))
    #visualize(point)
    return np.array(patch_list)


def check_num_points(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    num_points = np.array(pcd.points).shape[0]
    if num_points < 2048:
        print(ply_path,'  ', num_points)


def main():
    parser = argparse.ArgumentParser(description='Extract six FPS-centered KNN patches of 2048 points.')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--ann', default='all-result.txt')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    root = Path(args.data_root)
    for record in read_annotations(root / args.ann):
        relative = Path(*Path(record['video_path']).parts[1:]).with_suffix('.ply')
        pointcloud = root / 'pointcloud' / relative
        target = feature_dir(root, record['video_path'])
        target.mkdir(parents=True, exist_ok=True)
        patch = knn_patch(str(pointcloud))
        np.save(target / 'patch.npy', patch)
        print(target / 'patch.npy')


if __name__ == '__main__':
    main()

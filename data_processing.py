import numpy as np
import os
import pickle
from config import *


def get_path(path_dir):
    path_list = []
    if os.path.isdir(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.txt'):
                    path_list.append(file_absolute_path)
    return path_list


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
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


def extract_data(path_dir, dic, save_x, save_y):
    path_list = sorted(get_path(path_dir))
    label_list = [i.split('/')[-2] for i in path_list]
    label_list = [dic[i] for i in label_list]
    X = []
    for i in path_list:
        point_set = np.loadtxt(i, delimiter=',').astype(np.float32)  # (10000, 6)
        point = farthest_point_sample(point_set, 1024)  # (1024, 6), 获取最远距离的1024个点
        X.append(point)
    X = np.array(X)
    y = np.array(label_list)
    pickle.dump(X, open(save_x, 'wb'), protocol=4)
    pickle.dump(y, open(save_y, 'wb'), protocol=4)


if __name__ == '__main__':
    extract_data('/raid/yinghua/PCPrior/modelnet40', dic_modelnet40, '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl', '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl')



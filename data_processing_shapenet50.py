import os
import numpy as np
import pickle
import pandas as pd


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


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.pts'):
                    path_list.append(file_absolute_path)
    return path_list


def extract_data(path_dir, save_x, save_y):
    path_list = sorted(get_path(path_dir))
    path_lable_list = [i.replace('points', 'points_label').replace('.pts', '.seg') for i in path_list]
    big_label_list = [i.split('/')[-3] for i in path_list] #[02691156, 02691157, ...]
    X = []
    label_list = []
    for i in range(len(path_list)):
        big_label = big_label_list[i]
        point_set = np.loadtxt(path_list[i], delimiter=' ').astype(np.float32)
        label_set = np.loadtxt(path_lable_list[i], delimiter='\n').astype(np.int32)
        df = pd.DataFrame(point_set)
        df['label'] = label_set
        for label, pdf in df.groupby('label'):
            if len(pdf) > 128:
                points = pdf.to_numpy()[:, :-1]
                points = np.hstack((points, points))
                label_list.append(str(big_label)+'_'+str(label))
                points = farthest_point_sample(points, 128)
                X.append(points)
            else:
                points = pdf.to_numpy()[:, :-1]
                points = np.hstack((points, points))
                n = int(128/len(points)) + 1
                points = np.concatenate([points]*n, axis=0)
                label_list.append(str(big_label)+'_'+str(label))
                points = farthest_point_sample(points, 128)
                X.append(points)

    label_key = sorted(list(set(label_list)))
    dic = dict(zip(label_key, range(len(label_key))))
    label_list = [dic[i] for i in label_list]
    y = np.array(label_list)
    X = np.array(X)
    pickle.dump(X, open(save_x, 'wb'), protocol=4)
    pickle.dump(y, open(save_y, 'wb'), protocol=4)


extract_data('/raid/yinghua/PCPrior/shapenet', '/raid/yinghua/PCPrior/pkl_data/shapenet/X.pkl', '/raid/yinghua/PCPrior/pkl_data/shapenet/y.pkl')

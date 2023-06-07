import os
import numpy as np
import pickle


def farthest_point_sample(point, npoint):
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
                if file_absolute_path.endswith('.txt') and 'Annotations' in file_absolute_path and 'stairs' not in file_absolute_path:
                    path_list.append(file_absolute_path)
    return path_list


def extract_data(path_dir, save_x, save_y):

    path_list = get_path(path_dir)
    label_list = [i.split('/')[-1].split('_')[0] for i in path_list]
    dic = dict(zip(sorted(list(set(label_list))), range(len(set(label_list)))))
    label_list = [dic[i] for i in label_list]
    X = []
    for i in path_list:
        point_set = np.loadtxt(i, delimiter=' ').astype(np.float32)
        point = farthest_point_sample(point_set, 1024)
        X.append(point)
    X = np.array(X)
    y = np.array(label_list)
    pickle.dump(X, open(save_x, 'wb'), protocol=4)
    pickle.dump(y, open(save_y, 'wb'), protocol=4)


extract_data('/raid/yinghua/PCPrior/Stanford3dDataset_v1.2_Aligned_Version', '/raid/yinghua/PCPrior/pkl_data/s3dis/X.pkl', '/raid/yinghua/PCPrior/pkl_data/s3dis/y.pkl')




import numpy as np
import os


def apfd(error_idx_list, pri_idx_list):
    error_idx_list = list(error_idx_list)
    pri_idx_list = list(pri_idx_list)
    n = len(pri_idx_list)
    m = len(error_idx_list)
    TF_list = [pri_idx_list.index(i) for i in error_idx_list]
    apfd = 1 - sum(TF_list)*1.0 / (n*m) + 1 / (2*n)
    return apfd


def get_correct_n(y_pre, y):
    correct_n = 0
    for i in range(len(y)):
        if y_pre[i] == y[i]:
            correct_n += 1
    return correct_n


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    idx_miss_list.append(i)
    return idx_miss_list


def get_miss_lable(target_train_pre, target_test_pre, y_train, y_test):
    idx_miss_train_list = get_idx_miss_class(target_train_pre, y_train)
    idx_miss_test_list = get_idx_miss_class(target_test_pre, y_test)
    miss_train_label = [0]*len(y_train)
    for i in idx_miss_train_list:
        miss_train_label[i]=1
    miss_train_label = np.array(miss_train_label)

    miss_test_label = [0]*len(y_test)
    for i in idx_miss_test_list:
        miss_test_label[i]=1
    miss_test_label = np.array(miss_test_label)

    return miss_train_label, miss_test_label, idx_miss_test_list


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


def get_model_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.pt'):
                    path_list.append(file_absolute_path)
    return path_list


def effect_size_paired(x1, x2):
    diff = np.array(x1) - np.array(x2)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    return mean_diff / std_diff
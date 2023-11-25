import pickle
import torch
import argparse
import os
import json
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from sklearn.metrics import accuracy_score
from feature_extraction import get_uncertainty_feature, get_sapce_feature
from tensorflow.keras import models
from tensorflow.keras import layers

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

ap = argparse.ArgumentParser()

ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--path_target_model", type=str)
ap.add_argument("--path_target_model_train_pre", type=str)
ap.add_argument("--path_target_model_test_pre", type=str)
ap.add_argument("--path_train_point_mutants_feature", type=str)
ap.add_argument("--path_test_point_mutants_feature", type=str)
ap.add_argument("--path_save_res", type=str)
args = ap.parse_args()

path_x = args.path_x
path_y = args.path_y
path_target_model = args.path_target_model
path_target_model_train_pre = args.path_target_model_train_pre
path_target_model_test_pre = args.path_target_model_test_pre
path_train_point_mutants_feature = args.path_train_point_mutants_feature
path_test_point_mutants_feature = args.path_test_point_mutants_feature
path_save_res = args.path_save_res


def get_res_ratio_list(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = round(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))
        ratio = round(n_hit / len(idx_miss_list), 4)
        res_ratio_list.append(ratio)
    return res_ratio_list


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def DNN(x):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(x.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model


def main():
    # model = torch.load(path_target_model)
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)

    # point mutants feature
    train_point_mutants_feature = pickle.load(open(path_train_point_mutants_feature, 'rb'))
    test_point_mutants_feature = pickle.load(open(path_test_point_mutants_feature, 'rb'))

    # space_feature
    space_feature_train_x = get_sapce_feature(train_x)
    space_feature_test_x = get_sapce_feature(test_x)

    # pre_feature
    pre_feature_train_x = pickle.load(open(path_target_model_train_pre, 'rb'))
    pre_feature_test_x = pickle.load(open(path_target_model_test_pre, 'rb'))

    # uncertainty_feature
    uncertainty_feature_train_x = get_uncertainty_feature(pre_feature_train_x)
    uncertainty_feature_test_x = get_uncertainty_feature(pre_feature_test_x)

    concat_train_all_feature = np.hstack((space_feature_train_x, pre_feature_train_x, uncertainty_feature_train_x, train_point_mutants_feature))
    concat_test_all_feature = np.hstack((space_feature_test_x, pre_feature_test_x, uncertainty_feature_test_x, test_point_mutants_feature))

    target_train_pre = pre_feature_train_x.argsort()[:, -1]
    target_test_pre = pre_feature_test_x.argsort()[:, -1]

    print('train acc', accuracy_score(target_train_pre, train_y))
    print('test acc', accuracy_score(target_test_pre, test_y))

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    model = DNN(concat_train_all_feature)
    model.fit(concat_train_all_feature, miss_train_label, epochs=20, batch_size=32)
    y_concat_all = model.predict(concat_test_all_feature)
    y_concat_all = np.array([i[0] for i in y_concat_all])

    dnn_rank_idx = y_concat_all.argsort()[::-1].copy()
    select_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    idx_miss_list = get_idx_miss_class(target_test_pre, test_y)
    dnn_pfd = get_res_ratio_list(idx_miss_list, dnn_rank_idx, select_ratio_list)
    path_save = path_save_res.split('/')[-1].replace('.json', '')

    write_result(path_save+'->'+str(dnn_pfd), './result_dnn/result_dnn_pfd.txt')


if __name__ == '__main__':
    main()

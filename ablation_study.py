import pickle
import torch
import argparse
import json
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from feature_extraction import get_uncertainty_feature, get_sapce_feature

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


def main():
    model = torch.load(path_target_model)
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

    target_train_pre = pre_feature_train_x.argsort()[:, -1]
    target_test_pre = pre_feature_test_x.argsort()[:, -1]

    print('train acc', accuracy_score(target_train_pre, train_y))
    print('test acc', accuracy_score(target_test_pre, test_y))

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    dic = {}

    feature_without_train_point_mutants_feature = np.hstack((space_feature_train_x, pre_feature_train_x, uncertainty_feature_train_x))
    feature_without_test_point_mutants_feature = np.hstack((space_feature_test_x, pre_feature_test_x, uncertainty_feature_test_x))

    model = LGBMClassifier(n_estimators=300)
    model.fit(feature_without_train_point_mutants_feature, miss_train_label)
    y_concat_all = model.predict_proba(feature_without_test_point_mutants_feature)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()
    feature_without_point_mutants_feature_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
    dic['feature_without_point_mutants_feature_apfd'] = feature_without_point_mutants_feature_apfd

    feature_without_space_feature_train_x = np.hstack((pre_feature_train_x, uncertainty_feature_train_x, train_point_mutants_feature))
    feature_without_space_feature_test_x = np.hstack((pre_feature_test_x, uncertainty_feature_test_x, test_point_mutants_feature))

    model = LGBMClassifier(n_estimators=300)
    model.fit(feature_without_space_feature_train_x, miss_train_label)
    y_concat_all = model.predict_proba(feature_without_space_feature_test_x)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()
    feature_without_space_feature_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
    dic['feature_without_space_feature_apfd'] = feature_without_space_feature_apfd

    feature_without_pre_feature_train_x = np.hstack((space_feature_train_x, uncertainty_feature_train_x, train_point_mutants_feature))
    feature_without_pre_feature_test_x = np.hstack((space_feature_test_x, uncertainty_feature_test_x, test_point_mutants_feature))

    model = LGBMClassifier(n_estimators=300)
    model.fit(feature_without_pre_feature_train_x, miss_train_label)
    y_concat_all = model.predict_proba(feature_without_pre_feature_test_x)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()
    feature_without_pre_feature_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
    dic['feature_without_pre_feature_apfd'] = feature_without_pre_feature_apfd

    feature_without_uncertainty_feature_train_x = np.hstack((space_feature_train_x, pre_feature_train_x, train_point_mutants_feature))
    feature_without_uncertainty_feature_test_x = np.hstack((space_feature_test_x, pre_feature_test_x, test_point_mutants_feature))

    model = LGBMClassifier(n_estimators=300)
    model.fit(feature_without_uncertainty_feature_train_x, miss_train_label)
    y_concat_all = model.predict_proba(feature_without_uncertainty_feature_test_x)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()
    feature_without_uncertainty_feature_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
    dic['feature_without_uncertainty_feature_apfd'] = feature_without_uncertainty_feature_apfd

    json.dump(dic, open(path_save_res, 'w'), sort_keys=False, indent=4)
    print(dic)


if __name__ == '__main__':
    main()


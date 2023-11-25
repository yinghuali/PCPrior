import pickle
import torch
import argparse
import json
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def get_res_ratio_list(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = round(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))
        ratio = round(n_hit / len(idx_miss_list), 4)
        res_ratio_list.append(ratio)
    return res_ratio_list


def main():
    select_pfd =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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

    #pre_feature
    pre_feature_train_x = pickle.load(open(path_target_model_train_pre, 'rb'))
    pre_feature_test_x = pickle.load(open(path_target_model_test_pre, 'rb'))

    #uncertainty_feature
    uncertainty_feature_train_x = get_uncertainty_feature(pre_feature_train_x)
    uncertainty_feature_test_x = get_uncertainty_feature(pre_feature_test_x)

    concat_train_all_feature = np.hstack((space_feature_train_x, pre_feature_train_x, uncertainty_feature_train_x, train_point_mutants_feature))
    concat_test_all_feature = np.hstack((space_feature_test_x, pre_feature_test_x, uncertainty_feature_test_x, test_point_mutants_feature))

    target_train_pre = pre_feature_train_x.argsort()[:, -1]
    target_test_pre = pre_feature_test_x.argsort()[:, -1]

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    model = LGBMClassifier(n_estimators=300)
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()

    model = XGBClassifier(n_estimators=300)
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    xgb_rank_idx = y_concat_all.argsort()[::-1].copy()

    model = RandomForestClassifier()
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    rf_rank_idx = y_concat_all.argsort()[::-1].copy()

    model = LogisticRegression()
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    lr_rank_idx = y_concat_all.argsort()[::-1].copy()

    deepGini_rank_idx = DeepGini_rank_idx(pre_feature_test_x)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(pre_feature_test_x)
    pcs_rank_idx = PCS_rank_idx(pre_feature_test_x)
    entropy_rank_idx = Entropy_rank_idx(pre_feature_test_x)
    random_rank_idx = Random_rank_idx(pre_feature_test_x)

    select_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    idx_miss_list = get_idx_miss_class(target_test_pre, test_y)
    deepGini_pfd = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    random_pfd = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)
    vanillasoftmax_pfd = get_res_ratio_list(idx_miss_list, vanillasoftmax_rank_idx, select_ratio_list)
    pcs_pfd = get_res_ratio_list(idx_miss_list, pcs_rank_idx, select_ratio_list)
    entropy_pfd = get_res_ratio_list(idx_miss_list, entropy_rank_idx, select_ratio_list)
    
    lgb_pfd = get_res_ratio_list(idx_miss_list, lgb_rank_idx, select_ratio_list)
    xgb_pfd = get_res_ratio_list(idx_miss_list, xgb_rank_idx, select_ratio_list)
    rf_pfd = get_res_ratio_list(idx_miss_list, rf_rank_idx, select_ratio_list)
    lr_pfd = get_res_ratio_list(idx_miss_list, lr_rank_idx, select_ratio_list)


    dic = {
        'random_pfd': random_pfd,
        'deepGini_pfd': deepGini_pfd,
        'vanillasoftmax_pfd': vanillasoftmax_pfd,
        'pcs_pfd': pcs_pfd,
        'entropy_pfd': entropy_pfd,
        'lgb_pfd': lgb_pfd,
        'xgb_pfd': xgb_pfd,
        'rf_pfd': rf_pfd,
        'lr_pfd': lr_pfd,
    }

    json.dump(dic, open(path_save_res, 'w'), sort_keys=False, indent=4)

    print('random_pfd', random_pfd)
    print('deepGini_pfd', deepGini_pfd)
    print('vanillasoftmax_pfd', vanillasoftmax_pfd)
    print('pcs_pfd', pcs_pfd)
    print('entropy_pfd', entropy_pfd)
    print('lgb_pfd', lgb_pfd)
    print('xgb_pfd', xgb_pfd)
    print('rf_pfd', rf_pfd)
    print('lr_pfd', lr_pfd)


if __name__ == '__main__':
    main()






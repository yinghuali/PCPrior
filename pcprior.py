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

# python pcprior.py --path_x '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --path_target_model './target_models/modelnet40_pointnet_2.pt' --path_target_model_train_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_train_pre.pkl' --path_target_model_test_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_test_pre.pkl' --path_train_point_mutants_feature '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_train_point_mutants_feature_vec.pkl' --path_test_point_mutants_feature '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_test_point_mutants_feature_vec.pkl' --path_save_res './result/modelnet40_pointnet_2.json'


def main():
    model = torch.load(path_target_model)
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)

    # point mutants feature
    train_point_mutants_feature = pickle.load(open(path_train_point_mutants_feature, 'rb'))
    test_point_mutants_feature = pickle.load(open(path_test_point_mutants_feature, 'rb'))

    # pre_feature
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

    random_apfd = apfd(idx_miss_test_list, random_rank_idx)
    deepGini_apfd = apfd(idx_miss_test_list, deepGini_rank_idx)
    vanillasoftmax_apfd = apfd(idx_miss_test_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_test_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_test_list, entropy_rank_idx)

    lgb_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
    xgb_apfd = apfd(idx_miss_test_list, xgb_rank_idx)
    rf_apfd = apfd(idx_miss_test_list, rf_rank_idx)
    lr_apfd = apfd(idx_miss_test_list, lr_rank_idx)

    dic = {
        'random_apfd': random_apfd,
        'deepGini_apfd': deepGini_apfd,
        'vanillasoftmax_apfd': vanillasoftmax_apfd,
        'pcs_apfd': pcs_apfd,
        'entropy_apfd': entropy_apfd,
        'lgb_apfd': lgb_apfd,
        'xgb_apfd': xgb_apfd,
        'rf_apfd': rf_apfd,
        'lr_apfd': lr_apfd,
    }

    json.dump(dic, open(path_save_res, 'w'), sort_keys=False, indent=4)

    print('random_apfd', random_apfd)
    print('deepGini_apfd', deepGini_apfd)
    print('vanillasoftmax_apfd', vanillasoftmax_apfd)
    print('pcs_apfd', pcs_apfd)
    print('entropy_apfd', entropy_apfd)
    print('lgb_apfd', lgb_apfd)
    print('xgb_apfd', xgb_apfd)
    print('rf_apfd', rf_apfd)
    print('lr_apfd', lr_apfd)


if __name__ == '__main__':
    main()






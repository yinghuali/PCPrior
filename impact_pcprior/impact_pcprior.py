from feature_extraction import get_uncertainty_feature, get_sapce_feature
import pickle
import argparse
import json
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from utils import *

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

    max_depth_list = [1, 3, 5, 7, 9]
    colsample_bytree_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    learning_rate_list = [0.001, 0.01, 0.05, 0.1, 0.5]
    dic_depth = {}
    dic_colsample = {}
    dic_learning = {}
    for i in max_depth_list:
        model = LGBMClassifier(max_depth=i)
        model.fit(concat_train_all_feature, miss_train_label)
        y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
        lgb_rank_idx = y_concat_all.argsort()[::-1].copy()
        lgb_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
        print(i, lgb_apfd)
        dic_depth[i] = lgb_apfd
    json.dump(dic_depth, open(path_save_res + '_dic_depth.json', 'w'), sort_keys=False, indent=4)

    for i in colsample_bytree_list:
        model = LGBMClassifier(colsample_bytree=i)
        model.fit(concat_train_all_feature, miss_train_label)
        y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
        lgb_rank_idx = y_concat_all.argsort()[::-1].copy()
        lgb_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
        print(i, lgb_apfd)
        dic_colsample[i] = lgb_apfd
    json.dump(dic_colsample, open(path_save_res + '_dic_colsample.json', 'w'), sort_keys=False, indent=4)

    for i in learning_rate_list:
        model = LGBMClassifier(learning_rate=i)
        model.fit(concat_train_all_feature, miss_train_label)
        y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
        lgb_rank_idx = y_concat_all.argsort()[::-1].copy()
        lgb_apfd = apfd(idx_miss_test_list, lgb_rank_idx)
        print(i, lgb_apfd)
        dic_learning[i] = lgb_apfd
    json.dump(dic_learning, open(path_save_res+'_dic_learning.json', 'w'), sort_keys=False, indent=4)


if __name__ == '__main__':
    main()






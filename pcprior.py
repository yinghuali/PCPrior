from feature_extraction import get_uncertainty_feature, get_sapce_feature
import pickle
import torch
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from lightgbm import LGBMClassifier

path_x = '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl'
path_y = '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl'
path_target_model = './target_models/modelnet40_pointnet_2.pt'
path_target_model_train_pre = '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_train_pre.pkl'
path_target_model_test_pre = '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_test_pre.pkl'


def main():
    model = torch.load(path_target_model)
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)

    # pre_feature
    space_feature_train_x = get_sapce_feature(train_x)
    space_feature_test_x = get_sapce_feature(test_x)

    #pre_feature
    pre_feature_train_x = pickle.load(open(path_target_model_train_pre, 'rb'))
    pre_feature_test_x = pickle.load(open(path_target_model_test_pre, 'rb'))

    #uncertainty_feature
    uncertainty_feature_train_x = get_uncertainty_feature(pre_feature_train_x)
    uncertainty_feature_test_x = get_uncertainty_feature(pre_feature_test_x)


    concat_train_all_feature = np.hstack((space_feature_train_x, pre_feature_train_x, uncertainty_feature_train_x))
    concat_test_all_feature = np.hstack((space_feature_test_x, pre_feature_test_x, uncertainty_feature_test_x))


    target_train_pre = pre_feature_train_x.argsort()[:, -1]
    target_test_pre = pre_feature_test_x.argsort()[:, -1]

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    model = LGBMClassifier()
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()


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

    print('random_apfd', random_apfd)
    print('deepGini_apfd', deepGini_apfd)
    print('vanillasoftmax_apfd', vanillasoftmax_apfd)
    print('pcs_apfd', pcs_apfd)
    print('entropy_apfd', entropy_apfd)
    print('lgb_apfd', lgb_apfd)


if __name__ == '__main__':
    main()








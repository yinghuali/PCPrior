from feature_extraction import get_uncertainty_feature, get_sapce_feature
import pickle
import torch
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from torch import nn

path_x = '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl'
path_y = '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl'
path_target_model = './target_models/modelnet40_pointnet_5.pt'
path_target_model_pre = '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_5_train_pre.pkl'


def main():
    model = torch.load(path_target_model)
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)



    # pre_feature
    space_feature_train_x = get_sapce_feature(train_x)
    space_feature_test_x = get_sapce_feature(test_x)

    #pre_feature
    pre_feature = pickle.load(open(path_target_model_pre, 'rb'))
    pre_feature_train_x, pre_feature_test_x, _, _ = train_test_split(pre_feature, y, test_size=0.3, random_state=17)

    #uncertainty_feature
    uncertainty_feature = get_uncertainty_feature(pre_feature)
    uncertainty_feature_train_x, uncertainty_feature_test_x, _, _ = train_test_split(uncertainty_feature, y, test_size=0.3, random_state=17)

    concat_train_all_feature = np.hstack((space_feature_train_x, pre_feature_train_x, uncertainty_feature_train_x))
    concat_test_all_feature = np.hstack((space_feature_test_x, pre_feature_test_x, uncertainty_feature_test_x))

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, y_train,
                                                                           y_test)




    print('finished')


if __name__ == '__main__':
    main()








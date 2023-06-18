from feature_extraction import get_uncertainty_feature, get_sapce_feature
import pickle
import torch
import argparse
import json
from sklearn.model_selection import train_test_split
from get_rank_idx import *
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


target_model_path = './target_models/modelnet40_pointnet_2.pt'
path_x_np = '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl'
path_y = '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl'

path_target_model_train_pre = '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_train_pre.pkl'
path_target_model_test_pre = '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_test_pre.pk'

path_train_point_mutants_feature = '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_train_point_mutants_feature_vec.pkl'
path_test_point_mutants_feature = '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_test_point_mutants_feature_vec.pkl'



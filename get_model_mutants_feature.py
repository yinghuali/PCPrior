import numpy as np
import torch
import argparse
import pickle
from torch import nn
from utils import get_model_path
from sklearn.model_selection import train_test_split


ap = argparse.ArgumentParser()
ap.add_argument("--path_mutants_dir", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--save_train_model_mutants_feature_vec", type=str)
ap.add_argument("--save_test_model_mutants_feature_vec", type=str)
args = ap.parse_args()

# python get_model_mutants_feature.py --path_mutants_dir '/raid/yinghua/PCPrior/mutants/modelnet40_pointnet_2' --path_x '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --save_train_model_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_train_model_mutants_feature_vec.pkl' --save_test_model_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_test_model_mutants_feature_vec.pkl'
# nohup python get_model_mutants_feature.py --path_mutants_dir '/raid/yinghua/PCPrior/mutants/modelnet40_pointnet_2' --path_x '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --save_train_model_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_train_model_mutants_feature_vec.pkl' --save_test_model_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_test_model_mutants_feature_vec.pkl' > /dev/null 2>&1 &


def get_diff_feature(y, y_pre):
    vec = []
    for i in range(len(y)):
        if y[i] != y_pre[i]:
            vec.append(1)
        else:
            vec.append(0)
    return vec


def get_model_mutants_feature(path_mutants_dir, path_x, path_y, save_train_model_mutants_feature_vec, save_test_model_mutants_feature_vec):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model_path = sorted(get_model_path(path_mutants_dir))
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)

    all_feature_vec = []
    for i in range(len(model_path)):
        model = torch.load(model_path[i])
        model.to(device)
        all_pre_vec = []
        left = 0
        while left < len(train_x):
            train_select = train_x[left:left+16, ]
            x_train_t = torch.from_numpy(train_select).to(device)
            x_train_t = x_train_t.transpose(2, 1)
            with torch.no_grad():
                pred, trans_feat = model(x_train_t)
            probs = nn.Softmax(dim=1)(pred)
            pre_vec = probs.cpu().numpy()
            y_pre = list(pre_vec.argsort()[:, -1])
            all_pre_vec += y_pre
            left += 16
        all_pre_vec = np.array(all_pre_vec)
        feature_vec = get_diff_feature(train_y, all_pre_vec)
        all_feature_vec.append(feature_vec)
    all_feature_vec = np.array(all_feature_vec)
    train_model_mutants_feature_vec = all_feature_vec.T
    pickle.dump(train_model_mutants_feature_vec, open(save_train_model_mutants_feature_vec, 'wb'), protocol=4)

    all_feature_vec = []
    for i in range(len(model_path)):
        model = torch.load(model_path[i])
        model.to(device)
        all_pre_vec = []
        left = 0
        while left < len(test_x):
            test_select = test_x[left:left+16, ]
            x_test_t = torch.from_numpy(test_select).to(device)
            x_test_t = x_test_t.transpose(2, 1)
            with torch.no_grad():
                pred, trans_feat = model(x_test_t)
            probs = nn.Softmax(dim=1)(pred)
            pre_vec = probs.cpu().numpy()
            y_pre = list(pre_vec.argsort()[:, -1])
            all_pre_vec += y_pre
            left += 16
        all_pre_vec = np.array(all_pre_vec)
        feature_vec = get_diff_feature(test_y, all_pre_vec)
        all_feature_vec.append(feature_vec)
    all_feature_vec = np.array(all_feature_vec)
    test_model_mutants_feature_vec = all_feature_vec.T
    pickle.dump(test_model_mutants_feature_vec, open(save_test_model_mutants_feature_vec, 'wb'), protocol=4)


get_model_mutants_feature(args.path_mutants_dir,
                          args.path_x,
                          args.path_y,
                          args.save_train_model_mutants_feature_vec,
                          args.save_test_model_mutants_feature_vec
                          )


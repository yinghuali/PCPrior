import numpy as np
import torch
import argparse
import pickle
from torch import nn
from utils import get_model_path
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--path_target_model", type=str)
ap.add_argument("--path_x_all_mutants", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--save_train_point_mutants_feature_vec", type=str)
ap.add_argument("--save_test_point_mutants_feature_vec", type=str)
ap.add_argument("--path_target_train_pre", type=str)
ap.add_argument("--path_target_test_pre", type=str)
args = ap.parse_args()


# nohup python get_point_mutants_feature.py --path_target_model '/home/yinghua/pycharm/PCPrior/target_models/modelnet40_pointnet_2.pt' --path_x_all_mutants '/raid/yinghua/PCPrior/pkl_data/modelnet40/x_all_mutants.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --save_train_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_train_point_mutants_feature_vec.pkl' --save_test_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet_2_test_point_mutants_feature_vec.pkl' --path_target_train_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_train_pre.pkl' --path_target_test_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet_2_test_pre.pkl' > /dev/null 2>&1 &
# nohup python get_point_mutants_feature.py --path_target_model '/home/yinghua/pycharm/PCPrior/target_models/modelnet40_pointnet2_msg_2.pt' --path_x_all_mutants '/raid/yinghua/PCPrior/pkl_data/modelnet40/x_all_mutants.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --save_train_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet2_msg_2_train_point_mutants_feature_vec.pkl' --save_test_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet2_msg_2_test_point_mutants_feature_vec.pkl' --path_target_train_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet2_msg_2_train_pre.pkl' --path_target_test_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet2_msg_2_test_pre.pkl' > /dev/null 2>&1 &
# nohup python get_point_mutants_feature.py --path_target_model '/home/yinghua/pycharm/PCPrior/target_models/modelnet40_pointnet2_ssg_2.pt' --path_x_all_mutants '/raid/yinghua/PCPrior/pkl_data/modelnet40/x_all_mutants.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --save_train_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet2_ssg_2_train_point_mutants_feature_vec.pkl' --save_test_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40/pointnet2_ssg_2_test_point_mutants_feature_vec.pkl' --path_target_train_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet2_ssg_2_train_pre.pkl' --path_target_test_pre '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointnet2_ssg_2_test_pre.pkl' > /dev/null 2>&1 &

# nohup python get_point_mutants_feature.py --path_target_model '/home/yinghua/pycharm/PCPrior/target_models/shapenet_pointnet_0.pt' --path_x_all_mutants '/raid/yinghua/PCPrior/pkl_data/shapenet/x_all_mutants.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/shapenet/y.pkl' --save_train_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/shapenet/pointnet_0_train_point_mutants_feature_vec.pkl' --save_test_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/shapenet/pointnet_0_test_point_mutants_feature_vec.pkl' --path_target_train_pre '/raid/yinghua/PCPrior/pkl_data/shapenet_pre/shapenet_pre_pointnet_0_train_pre.pkl' --path_target_test_pre '/raid/yinghua/PCPrior/pkl_data/shapenet_pre/shapenet_pre_pointnet_0_test_pre.pkl' > /dev/null 2>&1 &
# nohup python get_point_mutants_feature.py --path_target_model '/home/yinghua/pycharm/PCPrior/target_models/shapenet_pointnet2_msg_0.pt' --path_x_all_mutants '/raid/yinghua/PCPrior/pkl_data/shapenet/x_all_mutants.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/shapenet/y.pkl' --save_train_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/shapenet/pointnet2_msg_0_train_point_mutants_feature_vec.pkl' --save_test_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/shapenet/pointnet2_msg_0_test_point_mutants_feature_vec.pkl' --path_target_train_pre '/raid/yinghua/PCPrior/pkl_data/shapenet_pre/shapenet_pre_pointnet2_msg_0_train_pre.pkl' --path_target_test_pre '/raid/yinghua/PCPrior/pkl_data/shapenet_pre/shapenet_pre_pointnet2_msg_0_test_pre.pkl' > /dev/null 2>&1 &
# nohup python get_point_mutants_feature.py --path_target_model '/home/yinghua/pycharm/PCPrior/target_models/shapenet_pointnet2_ssg_0.pt' --path_x_all_mutants '/raid/yinghua/PCPrior/pkl_data/shapenet/x_all_mutants.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/shapenet/y.pkl' --save_train_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/shapenet/pointnet2_ssg_0_train_point_mutants_feature_vec.pkl' --save_test_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/shapenet/pointnet2_ssg_0_test_point_mutants_feature_vec.pkl' --path_target_train_pre '/raid/yinghua/PCPrior/pkl_data/shapenet_pre/shapenet_pre_pointnet2_ssg_0_train_pre.pkl' --path_target_test_pre '/raid/yinghua/PCPrior/pkl_data/shapenet_pre/shapenet_pre_pointnet2_ssg_0_test_pre.pkl' > /dev/null 2>&1 &


# nohup python get_point_mutants_feature.py --path_target_model '/home/yinghua/pycharm/PCPrior/target_models/s3dis_pointnet_15.pt' --path_x_all_mutants '/raid/yinghua/PCPrior/pkl_data/s3dis/x_all_mutants.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/s3dis/y.pkl' --save_train_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/s3dis/pointnet_15_train_point_mutants_feature_vec.pkl' --save_test_point_mutants_feature_vec '/raid/yinghua/PCPrior/pkl_data/s3dis/pointnet_15_test_point_mutants_feature_vec.pkl' --path_target_train_pre '/raid/yinghua/PCPrior/pkl_data/s3dis_pre/s3dis_pre_pointnet_15_train_pre.pkl' --path_target_test_pre '/raid/yinghua/PCPrior/pkl_data/s3dis_pre/s3dis_pre_pointnet_15_test_pre.pkl' > /dev/null 2>&1 &


def get_diff_feature(target_y, y_pre):
    vec = []
    for i in range(len(target_y)):
        if target_y[i] != y_pre[i]:
            vec.append(1)
        else:
            vec.append(0)
    return vec


def get_point_mutants_feature(path_target_model, path_x_all_mutants, path_y, save_train_point_mutants_feature_vec, save_test_point_mutants_feature_vec, path_target_train_pre, path_target_test_pre):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(path_x_all_mutants, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    model = torch.load(path_target_model)
    model.to(device)

    target_train_pre = pickle.load(open(path_target_train_pre, 'rb'))
    target_test_pre = pickle.load(open(path_target_test_pre, 'rb'))
    target_train_pre = target_train_pre.argsort()[:, -1]
    target_test_pre = target_test_pre.argsort()[:, -1]

    all_feature_vec = []
    for x in data:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
        left = 0
        all_pre_vec = []
        while left < len(train_x):
            train_select = train_x[left:left+16, ]
            x_train_t = torch.from_numpy(train_select).to(device).float()
            x_train_t = x_train_t.transpose(2, 1)
            with torch.no_grad():
                pred, trans_feat = model(x_train_t)
            probs = nn.Softmax(dim=1)(pred)
            pre_vec = probs.cpu().numpy()
            y_pre = list(pre_vec.argsort()[:, -1])
            all_pre_vec += y_pre
            left += 16
        all_pre_vec = np.array(all_pre_vec)
        feature_vec = get_diff_feature(target_train_pre, all_pre_vec)
        all_feature_vec.append(feature_vec)

    all_feature_vec = np.array(all_feature_vec)
    train_point_mutants_feature_vec = all_feature_vec.T
    pickle.dump(train_point_mutants_feature_vec, open(save_train_point_mutants_feature_vec, 'wb'), protocol=4)

    all_feature_vec = []
    for x in data:
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
        left = 0
        all_pre_vec = []
        while left < len(test_x):
            test_select = test_x[left:left+16, ]
            x_test_t = torch.from_numpy(test_select).to(device).float()
            x_test_t = x_test_t.transpose(2, 1)
            with torch.no_grad():
                pred, trans_feat = model(x_test_t)
            probs = nn.Softmax(dim=1)(pred)
            pre_vec = probs.cpu().numpy()
            y_pre = list(pre_vec.argsort()[:, -1])
            all_pre_vec += y_pre
            left += 16
        all_pre_vec = np.array(all_pre_vec)
        feature_vec = get_diff_feature(target_test_pre, all_pre_vec)
        all_feature_vec.append(feature_vec)

    all_feature_vec = np.array(all_feature_vec)
    test_point_mutants_feature_vec = all_feature_vec.T
    pickle.dump(test_point_mutants_feature_vec, open(save_test_point_mutants_feature_vec, 'wb'), protocol=4)


if __name__ == '__main__':
    get_point_mutants_feature(args.path_target_model,
                              args.path_x_all_mutants,
                              args.path_y,
                              args.save_train_point_mutants_feature_vec,
                              args.save_test_point_mutants_feature_vec,
                              args.path_target_train_pre,
                              args.path_target_test_pre
                              )


import pickle
import torch
import provider
from utils import *
from sklearn.model_selection import train_test_split
from torch import nn

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--cuda", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--model_path", type=str)
ap.add_argument("--save_train_vec", type=str)
ap.add_argument("--save_test_vec", type=str)
args = ap.parse_args()

# python get_model_pointconv_pre.py --path_x '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl'  --model_path './target_models/modelnet40_pointconv_8.pt' --save_train_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointconv_8_train_pre.pkl' --save_test_vec '/raid/yinghua/PCPrior/pkl_data/modelnet40_pre/modelnet40_pre_pointconv_8_test_pre.pkl'


def main():

    model = torch.load(args.model_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = pickle.load(open(args.path_x, 'rb'))
    y = pickle.load(open(args.path_y, 'rb'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)

    all_correct_n = 0
    all_pre_vec = []

    left = 0
    while left < len(train_x):
        train_select = train_x[left:left+16, ]
        train_select_y = train_y[left:left + 16]
        x_train_t = train_select
        jittered_data = provider.random_scale_point_cloud(x_train_t[:, :, 0:3], scale_low=2.0 / 3, scale_high=3 / 2.0)
        jittered_data = provider.shift_point_cloud(jittered_data, shift_range=0.2)
        x_train_t[:, :, 0:3] = jittered_data
        x_train_t = provider.random_point_dropout_v2(x_train_t)
        provider.shuffle_points(x_train_t)
        x_train_t = torch.Tensor(x_train_t)

        x_train_t = x_train_t.transpose(2, 1)
        x_train_t = x_train_t.to(device)

        with torch.no_grad():
            pred = model(x_train_t[:, :3, :], x_train_t[:, 3:, :])

        probs = nn.Softmax(dim=1)(pred)

        pre_vec = probs.cpu().numpy()
        all_pre_vec.append(pre_vec)

        y_pre = pre_vec.argsort()[:, -1]
        correct_n = get_correct_n(y_pre, train_select_y)
        all_correct_n += correct_n
        left += 16

    all_pre_vec = np.concatenate(all_pre_vec, axis=0)
    pickle.dump(all_pre_vec, open(args.save_train_vec, 'wb'), protocol=4)
    print('train_acc=', all_correct_n*1.0/len(train_x))

    all_correct_n = 0
    all_pre_vec = []

    left = 0
    while left < len(test_x):
        test_select = test_x[left:left+16, ]
        test_select_y = test_y[left:left + 16]
        x_test_t = test_select
        jittered_data = provider.random_scale_point_cloud(x_test_t[:, :, 0:3], scale_low=2.0 / 3, scale_high=3 / 2.0)
        jittered_data = provider.shift_point_cloud(jittered_data, shift_range=0.2)
        x_test_t[:, :, 0:3] = jittered_data
        x_test_t = provider.random_point_dropout_v2(x_test_t)
        provider.shuffle_points(x_test_t)
        x_test_t = torch.Tensor(x_test_t)

        x_test_t = x_test_t.transpose(2, 1)
        x_test_t = x_test_t.to(device)

        with torch.no_grad():
            pred = model(x_test_t[:, :3, :], x_test_t[:, 3:, :])

        probs = nn.Softmax(dim=1)(pred)

        pre_vec = probs.cpu().numpy()
        all_pre_vec.append(pre_vec)

        y_pre = pre_vec.argsort()[:, -1]
        correct_n = get_correct_n(y_pre, test_select_y)
        all_correct_n += correct_n
        left += 16

    all_pre_vec = np.concatenate(all_pre_vec, axis=0)
    pickle.dump(all_pre_vec, open(args.save_test_vec, 'wb'), protocol=4)
    print('test_acc=', all_correct_n * 1.0 / len(test_x))


if __name__ == '__main__':
    main()


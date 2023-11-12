import torch
import json
import torch.nn.functional as F
from get_rank_idx import *
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from torch import nn
import pickle
import torch
import argparse
from models.pointnet_cls import get_model
from torch import nn, optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from feature_extraction import get_uncertainty_feature, get_sapce_feature


ap = argparse.ArgumentParser()
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--path_target_model", type=str)
ap.add_argument("--path_target_model_train_pre", type=str)
ap.add_argument("--path_target_model_test_pre", type=str)
ap.add_argument("--path_train_point_mutants_feature", type=str)
ap.add_argument("--path_test_point_mutants_feature", type=str)
ap.add_argument("--path_save", type=str)
args = ap.parse_args()

path_target_model = args.path_target_model
path_x = args.path_x
path_y = args.path_y

path_target_model_train_pre = args.path_target_model_train_pre
path_target_model_test_pre = args.path_target_model_test_pre

path_train_point_mutants_feature = args.path_train_point_mutants_feature
path_test_point_mutants_feature = args.path_test_point_mutants_feature
path_save = args.path_save


ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
epochs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
x = pickle.load(open(path_x, 'rb'))
y = pickle.load(open(path_y, 'rb'))
train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
train_x, candidate_train_x, train_y, candidate_y = train_test_split(train_x_, train_y_, test_size=0.5, random_state=17)

pre_feature_train_x_ = pickle.load(open(path_target_model_train_pre, 'rb'))
pre_feature_train_x, candidate_pre_feature_train_x, _, _ = train_test_split(pre_feature_train_x_, train_y_, test_size=0.5, random_state=17)

idx_np = np.array(list(range(len(x))))
train_all_idx, test_all_idx_, _, _ = train_test_split(idx_np, y, test_size=0.3, random_state=17)
train_idx, candidate_idx, _, _ = train_test_split(train_all_idx, train_y_, test_size=0.5, random_state=17)


def get_compared_idx():
    deepGini_rank_idx = DeepGini_rank_idx(candidate_pre_feature_train_x)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(candidate_pre_feature_train_x)
    pcs_rank_idx = PCS_rank_idx(candidate_pre_feature_train_x)
    entropy_rank_idx = Entropy_rank_idx(candidate_pre_feature_train_x)

    # mp_rank_idx = MP_rank_idx(candidate_pre_feature_train_x)
    # leastconfidence_rank_idx = LeastConfidence_rank_idx(candidate_pre_feature_train_x)
    # margin_rank_idx = Margin_rank_idx(candidate_pre_feature_train_x)

    random_rank_idx = Random_rank_idx(candidate_pre_feature_train_x)

    return deepGini_rank_idx, vanillasoftmax_rank_idx, pcs_rank_idx, entropy_rank_idx, random_rank_idx


def get_PCPrior_rank_idx():
    # point mutants feature
    train_point_mutants_feature_ = pickle.load(open(path_train_point_mutants_feature, 'rb'))
    train_point_mutants_feature, candidate_train_point_mutants_feature, _, _ = train_test_split(train_point_mutants_feature_, train_y_, test_size=0.5, random_state=17)
    test_point_mutants_feature = pickle.load(open(path_test_point_mutants_feature, 'rb'))

    # space_feature
    space_feature_train_x = get_sapce_feature(train_x)
    candidate_space_feature_train_x = get_sapce_feature(candidate_train_x)
    space_feature_test_x = get_sapce_feature(test_x)

    # pre_feature
    pre_feature_train_x_ = pickle.load(open(path_target_model_train_pre, 'rb'))
    pre_feature_train_x, candidate_pre_feature_train_x, _, _ = train_test_split(pre_feature_train_x_, train_y_,test_size=0.5, random_state=17)
    pre_feature_test_x = pickle.load(open(path_target_model_test_pre, 'rb'))

    # uncertainty_feature
    uncertainty_feature_train_x = get_uncertainty_feature(pre_feature_train_x)
    candidate_uncertainty_feature_train_x = get_uncertainty_feature(candidate_pre_feature_train_x)
    uncertainty_feature_test_x = get_uncertainty_feature(pre_feature_test_x)

    concat_train_all_feature = np.hstack((space_feature_train_x, pre_feature_train_x, uncertainty_feature_train_x, train_point_mutants_feature))
    concat_candidate_all_feature = np.hstack((candidate_space_feature_train_x, candidate_pre_feature_train_x, candidate_uncertainty_feature_train_x, candidate_train_point_mutants_feature))
    concat_test_all_feature = np.hstack((space_feature_test_x, pre_feature_test_x, uncertainty_feature_test_x, test_point_mutants_feature))

    target_train_pre = pre_feature_train_x.argsort()[:, -1]
    candidate_target_train_pre = candidate_pre_feature_train_x.argsort()[:, -1]
    target_test_pre = pre_feature_test_x.argsort()[:, -1]

    print('train acc', accuracy_score(target_train_pre, train_y))
    print('candidate acc', accuracy_score(candidate_target_train_pre, candidate_y))
    print('test acc', accuracy_score(target_test_pre, test_y))

    miss_train_label, miss_test_label, idx_miss_test_list = get_miss_lable(target_train_pre, target_test_pre, train_y, test_y)

    model = LGBMClassifier(n_estimators=300)
    model.fit(concat_train_all_feature, miss_train_label)
    y_concat_all = model.predict_proba(concat_candidate_all_feature)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()

    return lgb_rank_idx


def model_pre(x, y, model):
    all_correct_n = 0
    all_pre_vec = []
    left = 0
    model.eval()
    while left < len(x):
        train_select = x[left:left+16, ]
        train_select_y = y[left:left + 16]
        x_train_t = torch.from_numpy(train_select).to(device).float()
        x_train_t = x_train_t.transpose(2, 1)

        with torch.no_grad():
            pred, trans_feat = model(x_train_t)

        probs = nn.Softmax(dim=1)(pred)

        pre_vec = probs.cpu().numpy()
        all_pre_vec.append(pre_vec)

        y_pre = pre_vec.argsort()[:, -1]
        correct_n = get_correct_n(y_pre, train_select_y)
        all_correct_n += correct_n
        left += 16

    acc = all_correct_n*1.0/len(x)
    return acc


def get_retrain(rank_list):
    all_res = []
    for _ in range(10):
        model = torch.load(path_target_model)
        model = model.to(device)
        train_params = model.parameters()
        optimizer = optim.SGD(train_params, lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

        acc_list = []

        for i in range(len(ratio_list)):
            model.train()

            tmp_idx = rank_list[:int(len(rank_list) * ratio_list[i])]
            select_idx = list(train_idx) + list(tmp_idx)

            select_x_train = x[select_idx]
            select_y_train = y[select_idx]

            x_train_t = torch.from_numpy(select_x_train).float()
            y_train_t = torch.from_numpy(select_y_train)

            dataset = Data.TensorDataset(x_train_t, y_train_t)
            trainDataLoader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)

            for e in range(epochs_list[i]):
                for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                    if points.size(0) == 1:
                        continue
                    points = points.transpose(2, 1)
                    points = Variable(points, requires_grad=True).to(device)
                    target = Variable(target).to(device)
                    optimizer.zero_grad()
                    pred, trans_feat = model(points)
                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step()

            acc = model_pre(test_x, test_y, model)
            acc_list.append(acc)
            print('=====', acc, '======')

        all_res.append(acc_list)

    all_res = np.array(all_res)
    # all_res = np.mean(all_res, axis=0)
    all_res = np.max(all_res, axis=0)
    print('===========')
    print(all_res)
    print('===========')
    return list(all_res)


def main():

    deepGini_rank_idx, vanillasoftmax_rank_idx, pcs_rank_idx, entropy_rank_idx, random_rank_idx = get_compared_idx()
    lgb_rank_idx = get_PCPrior_rank_idx()

    deepGini_res = get_retrain(deepGini_rank_idx)
    vanillasoftmax_res = get_retrain(vanillasoftmax_rank_idx)
    pcs_res = get_retrain(pcs_rank_idx)
    entropy_res = get_retrain(entropy_rank_idx)
    random_res = get_retrain(random_rank_idx)
    lgb_res = get_retrain(lgb_rank_idx)

    dic = {
        'random_res': random_res,
        'deepGini_res': deepGini_res,
        'vanillasoftmax_res': vanillasoftmax_res,
        'pcs_res': pcs_res,
        'entropy_res': entropy_res,
        'lgb_res': lgb_res,
    }

    json.dump(dic, open(path_save, 'w'), sort_keys=False, indent=4)


if __name__ == '__main__':
    main()


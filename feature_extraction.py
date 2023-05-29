import numpy as np
import torch
import pickle

from scipy.stats import entropy


def get_sapce_feature(x):
    space_feature = []
    for i in x:
        space_feature.append(i.mean(axis=0))
    space_feature = np.array(space_feature)
    return space_feature


def get_uncertainty_feature(x):
    margin_score = np.sort(x)[:, -1] - np.sort(x)[:, -2]
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    least_score = x.max(1)
    VanillaSoftmax_score = 1 - x.max(1)
    PCS_score = 1 - (np.sort(x)[:, -1] - np.sort(x)[:, -2])
    entropy_score = entropy(np.array([i / np.sum(i) for i in x]), axis=1)

    feature_vec = np.vstack((margin_score, gini_score, least_score, VanillaSoftmax_score, PCS_score, entropy_score))
    return feature_vec.T


def get_mutants_point_feaure(path_x_all_mutants, path_model):
    model = torch.load(path_model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_all_mutants = pickle.load(open(path_x_all_mutants, 'rb'))
    print(x_all_mutants.shape)


get_mutants_point_feaure('/raid/yinghua/PCPrior/pkl_data/modelnet40/x_all_mutants.pkl', './target_models/modelnet40_pointnet_2.pt')




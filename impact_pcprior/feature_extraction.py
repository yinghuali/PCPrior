import numpy as np
from scipy.stats import entropy
from scipy.stats import skew, kurtosis


def get_sapce_feature(x):
    space_feature = []
    for i in x:
        var = i.var(axis=0)
        std = i.std(axis=0)
        mean = i.mean(axis=0)
        median = np.median(i, axis=0)
        diff = np.array([i[:, j].max() - i[:, j].min() for j in range(len(i[0]))])
        sk = skew(i, axis=0)
        ku = kurtosis(i, axis=0)
        mi = i.min(axis=0)
        ma = i.max(axis=0)
        p_q1 = np.percentile(i, 25, axis=0)
        p_q3 = np.percentile(i, 75, axis=0)
        abs = np.abs(i.mean(axis=0) - i).mean(axis=0)
        feature = np.hstack((var, std, mean, median, diff, sk, ku, mi, ma, p_q1, p_q3, abs))
        space_feature.append(feature)

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



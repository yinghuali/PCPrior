import torch
import random
import numpy as np
import argparse
ap = argparse.ArgumentParser()

ap.add_argument("--target_model_path", type=str)
ap.add_argument("--save_model_dir", type=str)
ap.add_argument("--n_mutants", type=int)
args = ap.parse_args()

# python get_model_mutants.py --target_model_path '/home/yinghua/pycharm/PCPrior/target_models/modelnet40_pointnet_2.pt' --save_model_dir '/raid/yinghua/PCPrior/mutants/modelnet40_pointnet_2' --n_mutants 100


def get_model_mutants(target_model_path, save_model_path):
    path_target_model = target_model_path
    model = torch.load(path_target_model, map_location='cpu')

    param = model.state_dict()
    key_list = list(param.keys())

    new_key_list = []
    for key in key_list:
        if len(param[key].shape) == 2:
            new_key_list.append(key)

    select_key = random.sample(new_key_list, 1)[0]
    row_n = param[select_key].shape[0]
    col_n = param[select_key].shape[1]
    try:
        select_row = random.sample(range(row_n), 100)
        select_col = random.sample(range(col_n), 100)
        for i in range(len(select_row)):
            model.state_dict()[select_key][select_row[i]][select_col[i]] += random.uniform(-1, 1)
    except:
        pass
    torch.save(model, save_model_path)


def main():
    for i in range(args.n_mutants):
        save_path = args.save_model_dir + '/' + str(i) + '.pt'
        get_model_mutants(args.target_model_path, save_path)


if __name__ == '__main__':
    main()









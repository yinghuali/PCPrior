import pickle
import random
import numpy as np
import argparse
ap = argparse.ArgumentParser()

ap.add_argument("--path_x", type=str)
ap.add_argument("--save_path", type=str)
args = ap.parse_args()


# nohup python get_point_mutants.py --path_x '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl' --save_path '/raid/yinghua/PCPrior/pkl_data/modelnet40/x_all_mutants.pkl' > /dev/null 2>&1 &
# nohup python get_point_mutants.py --path_x '/raid/yinghua/PCPrior/pkl_data/shapenet/X.pkl' --save_path '/raid/yinghua/PCPrior/pkl_data/shapenet/x_all_mutants.pkl' > /dev/null 2>&1 &
# nohup python get_point_mutants.py --path_x '/raid/yinghua/PCPrior/pkl_data/s3dis/X.pkl' --save_path '/raid/yinghua/PCPrior/pkl_data/s3dis/x_all_mutants.pkl' > /dev/null 2>&1 &

def get_mutation_point_feature(x, mutation_ratio, n_mutants):
    all_mutants = []
    for _ in range(n_mutants):
        single_mutants = []
        for i in range(len(x)):
            points = np.copy(x[i])
            select_points_idx = random.sample(range(len(points)), int(len(points) * mutation_ratio))
            for idx in select_points_idx:
                points[idx] += np.hstack((-1 + 2 * np.random.random((3)), np.array([0, 0, 0])))
                points[idx] = np.around(points[idx], 4)
            single_mutants.append(points)
        single_mutants = np.array(single_mutants)
        all_mutants.append(single_mutants)
    all_mutants = np.array(all_mutants)
    return all_mutants


if __name__ == '__main__':

    x = pickle.load(open(args.path_x, 'rb'))
    all_mutants = get_mutation_point_feature(x, 0.3, 30)
    pickle.dump(all_mutants, open(args.save_path, 'wb'), protocol=4)



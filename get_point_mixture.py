import pickle
import random
import numpy as np
import argparse
ap = argparse.ArgumentParser()

ap.add_argument("--path_x", type=str)
ap.add_argument("--save_path", type=str)
args = ap.parse_args()


def get_mixture_x(path_x, save_path):
    x = pickle.load(open(path_x, 'rb'))
    mixture_x = []
    for i in range(len(x)):
        points = np.copy(x[i])
        select_points_idx = random.sample(range(len(points)), int(len(points) * 0.3))
        for idx in select_points_idx:
            points[idx] += np.hstack((-1 + 2 * np.random.random((3)), np.array([0, 0, 0])))
            points[idx] = np.around(points[idx], 4)
        mixture_x.append(points)
    mixture_x = np.array(mixture_x)
    pickle.dump(mixture_x, open(save_path, 'wb'), protocol=4)


if __name__ == '__main__':
    get_mixture_x(args.path_x, args.save_path)




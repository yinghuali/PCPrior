import pickle
import torch
import argparse
import provider
from models.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from torch import nn, optim
from tqdm import tqdm
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser()

ap.add_argument("--epochs", type=int)
ap.add_argument("--k", type=int)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--batch_size", type=int)
ap.add_argument("--save_model_path", type=str)
args = ap.parse_args()

# python train_pointconv.py --k 40 --epochs 11 --batch_size 16  --path_x '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --save_model_path './target_models/modelnet40_pointconv'
# python train_pointconv.py --k 50 --epochs 10 --batch_size 16  --path_x '/raid/yinghua/PCPrior/pkl_data/shapenet/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/shapenet/y.pkl' --save_model_path './target_models/shapenet_pointconv'
# python train_pointconv.py --k 13 --epochs 11 --batch_size 16  --path_x '/raid/yinghua/PCPrior/pkl_data/s3dis/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/s3dis/y.pkl' --save_model_path './target_models/s3dis_pointconv'

def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")

    x = pickle.load(open(args.path_x, 'rb'))
    y = pickle.load(open(args.path_y, 'rb'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)

    x_train_t = torch.from_numpy(train_x).float()
    y_train_t = torch.from_numpy(train_y)

    x_test_t = torch.from_numpy(test_x).float()
    y_test_t = torch.from_numpy(test_y)

    dataset = Data.TensorDataset(x_train_t, y_train_t)
    trainDataLoader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    classifier = PointConvClsSsg(num_classes=args.k)
    classifier.to(device)
    train_params = classifier.parameters()
    optimizer = optim.SGD(train_params, lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for e in range(args.epochs):
        classifier = classifier.train()
        scheduler.step()
        all_correct = 0
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):

            points = points.data.numpy()
            jittered_data = provider.random_scale_point_cloud(points[:,:, 0:3], scale_low=2.0/3, scale_high=3/2.0)
            jittered_data = provider.shift_point_cloud(jittered_data, shift_range=0.2)
            points[:, :, 0:3] = jittered_data
            points = provider.random_point_dropout_v2(points)
            provider.shuffle_points(points)
            points = torch.Tensor(points)

            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)

            optimizer.zero_grad()
            pred = classifier(points[:, :3, :], points[:, 3:, :])

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            all_correct += correct

        print('all_correct=', all_correct)
        epoch_acc = all_correct*1.0 / len(train_x)
        print('epoch_acc =', epoch_acc)

        torch.save(classifier, args.save_model_path + '_' + str(e) + '.pt')


if __name__ == '__main__':
    main()


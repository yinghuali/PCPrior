import pickle
import torch
import argparse
from models.pointnet_cls import get_model
from torch import nn, optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser()

ap.add_argument("--epochs", type=int)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--batch_size", type=int)
ap.add_argument("--save_model_path", type=str)
args = ap.parse_args()

# python train_pointnet_cls.py --epochs 26 --batch_size 16  --path_x '/raid/yinghua/PCPrior/pkl_data/modelnet40/X.pkl' --path_y '/raid/yinghua/PCPrior/pkl_data/modelnet40/y.pkl' --save_model_path './target_models/modelnet40_pointnet'


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

x = pickle.load(open(args.path_x, 'rb'))
y = pickle.load(open(args.path_y, 'rb'))
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)


x_train_t = torch.from_numpy(train_x)
y_train_t = torch.from_numpy(train_y)

x_test_t = torch.from_numpy(test_x)
y_test_t = torch.from_numpy(test_y)

dataset = Data.TensorDataset(x_train_t, y_train_t)
trainDataLoader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

classifier = get_model(k=40)
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
    i = 0
    for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        points = points.transpose(2, 1)
        points = Variable(points, requires_grad=True).to(device)
        target = Variable(target).to(device)

        optimizer.zero_grad()
        pred, trans_feat = classifier(points)  # pred: [8, 40], 8 是batch size， 40是样本的类别总数
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        all_correct += correct

    print('all_correct=', all_correct)
    epoch_acc = all_correct*1.0 / len(train_x)
    print('epoch_acc =', epoch_acc)

    if e % 5 == 0 and e > 0:
        torch.save(classifier, args.save_model_path + '_' + str(e) + '.pt')








import torch
import torchvision
import torch.nn as nn
import numpy as np
import baseline
import densenet201
import densenet
import os
from sklearn import metrics
from MURAdataset import per_class_dataset

def get_labels():
    batch_size = 4
    test_transform = baseline.test_transform
    testset = per_class_dataset(csv_file=r'/valid.csv', root_dir=present_dir_path,
                                transform=test_transform, RGB=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    all_labels = np.array([])
    for data in testloader:
        images, labels, weights = data
        all_labels = np.append(all_labels, labels.numpy())
    return all_labels

def get_predicts(model, testloader):
    tot_predict = np.array([])
    for data in testloader:
        images, labels, weights = data
        # images = images.cuda()
        # labels = labels.cuda()
        outputs = (model(images))
        outputs = outputs.select(1, 0)
        outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
        # tot_predict = np.append(tot_predict, outputs.cpu().detach().numpy())
        tot_predict = np.append(tot_predict, outputs.detach().numpy())

    return tot_predict

def get_from_baseline(load_path):
    batch_size = 4
    test_transform = baseline.test_transform
    testset = per_class_dataset(csv_file=r'/valid.csv', root_dir=present_dir_path,
                                 transform=test_transform, RGB=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    net = torchvision.models.densenet169(pretrained=False)
    net.classifier = nn.Linear(26624, 1)
    net.load_state_dict(torch.load(load_path))

    # if torch.cuda.is_available():
    #     net = net.cuda()

    predicts = get_predicts(net, testloader)
    print ("finish model: %s" % load_path)
    return predicts

def get_from_densenet169(load_path):
    batch_size = 8
    test_transform = densenet.test_transform
    testset = per_class_dataset(csv_file=r'/valid.csv', root_dir=present_dir_path,
                                 transform=test_transform, RGB=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    net = torchvision.models.densenet169(pretrained=False)
    net.classifier = nn.Linear(1664, 1)
    net.load_state_dict(torch.load(load_path))

    # if torch.cuda.is_available():
    #     net = net.cuda()

    predicts = get_predicts(net, testloader)
    print ("finish model: %s" % load_path)
    return predicts

def get_from_densenet201(load_path):
    batch_size = 4
    test_transform = densenet201.test_transform
    testset = per_class_dataset(csv_file=r'/valid.csv', root_dir=present_dir_path,
                                 transform=test_transform, RGB=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    net = torchvision.models.densenet201(pretrained=False)
    net.classifier = nn.Linear(1920, 1)
    net.load_state_dict(torch.load(load_path))

    # if torch.cuda.is_available():
    #     net = net.cuda()

    predicts = get_predicts(net, testloader)
    print ("finish model: %s" % load_path)
    return predicts

def test(predicts):
    predicts = np.array(predicts)
    predicts = np.mean(predicts, axis=0,keepdims=False)
    predicts = torch.tensor(predicts)
    predicts = torch.sigmoid(predicts).numpy()
    auc = metrics.roc_auc_score(labels, predicts)
    print ("total AUC = %.3lf" % auc)

if __name__ == '__main__':
    present_dir_path = os.getcwd()
    labels = get_labels()

    predicts = get_from_baseline('./save/densenet169/baseline.pkl')
    predicts = [predicts]
    test(predicts)

    predicts = get_from_densenet169('./save/densenet169/b8_lr1e-4_d0_logloss_true.pkl')
    predicts = [predicts]
    test(predicts)

    predicts = get_from_densenet169('./save/densenet169/b8_lr1e-4_d0_focalloss.pkl')
    predicts = [predicts]
    test(predicts)

    predicts = get_from_densenet201('./save/densenet201/b8_lr1e-4_d0_logloss.pkl')
    predicts = [predicts]
    test(predicts)

    predicts = get_from_densenet201('./save/densenet201/b8_lr1e-4_d0_focalloss.pkl')
    predicts = [predicts]
    test(predicts)

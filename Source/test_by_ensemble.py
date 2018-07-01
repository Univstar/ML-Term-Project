import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MURAdataset import _load_model, clip, Focal_Loss, MURAdataset, per_class_dataset, sigmoid_test
import os
import sys
import argparse
import time
from sklearn import metrics
import numpy as np

if __name__ == '__main__':

    test_transform = transforms.Compose([transforms.Resize((320, 320)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])

    present_dir_path = os.getcwd()

    testset = per_class_dataset(csv_file='/valid.csv', root_dir=present_dir_path,
                                transform=test_transform, RGB=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=1)

    net = torchvision.models.densenet169(pretrained=False)
    net.classifier = nn.Linear(1664, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=True)

    if torch.cuda.device_count() > 1:
        print("Let's use GPUS!")
        net = nn.DataParallel(net)

    if torch.cuda.is_available():
        net = net.cuda()

    load_paths = [r"./save/densenet169/mura3_2noweight_acc.pkl", \
                  r"./save/densenet169/mura3_2noweight_loss.pkl", \
                  r"./save/densenet169/mura3cropfocal_1_acc.pkl", \
                  r"./save/densenet169/mura3cropfocal_1_loss.pkl"]

    tot_label, tot_predict = np.array([]), np.array([])

    for i in range(len(load_paths)):
        predict = np.array([])
        net.load_state_dict(torch.load(load_paths[i]))
        model = net
        print ("load from " + load_paths[i])
        for data in testloader:
            images, labels, weights = data
            images = images.cuda()
            labels = labels.cuda()
            weights = weights.cuda().float()
            outputs = torch.sigmoid(model(images))
            outputs = outputs.select(1, 0)
            outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)

            if i == 0:
                tot_label = np.append(tot_label, labels.cpu().numpy())
            predict = np.append(predict, outputs.cpu().detach().numpy())

        if i == 0:
            tot_predict = predict
        else:
            tot_predict += predict

    tot_predict /= len(load_paths)
    print ("AUC = %.3lf" % (metrics.roc_auc_score(tot_label, tot_predict)))
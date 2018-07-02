import copy
import numpy as np
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
import densenet

def make_Difficulty():
    load_model = True
    batch_size = 8
    if load_model == True:
        load_path = "./save/densenet169/mura3cropfocal_1_acc.pkl"

    train_transform = transforms.Compose([transforms.Resize((320, 320)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])
    net = torchvision.models.densenet169(pretrained=False)
    net.classifier = nn.Linear(1664, 1)
    net.load_state_dict(torch.load(load_path))
    if torch.cuda.is_available():
        net = net.cuda()

    present_dir_path = os.getcwd()
    trainset = per_class_dataset(csv_file=r'/train.csv', root_dir=present_dir_path,
                          transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    tot_loss = np.array([])
    net.eval()
    print ("Start to evaluate")
    for (i, data) in enumerate(trainloader, 0):
        print ("step %d" % i)
        images, labels, weights = data
        images = images.cuda()
        labels = labels.cuda()
        weights = weights.cuda().float()
        outputs = torch.sigmoid(net(images))
        outputs = outputs.select(1, 0)
        outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
        loss = -(labels.float() * outputs.log() + (1 - labels.float()) * (1 - outputs).log())
        loss *= weights
        tot_loss = np.append(tot_loss, loss.cpu().detach().numpy())

    Difficulty = np.argsort(tot_loss)
    np.save("./result/easy_to_hard.npy", Difficulty)

def check():
    Difficulty = np.load("./result/easy_to_hard.npy")
    for i in range(Difficulty.shape[0]):
        if i not in Difficulty:
            print (i)

if __name__ == '__main__':
    check()
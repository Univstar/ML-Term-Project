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
import densenet

if __name__ == '__main__':
    load_model = True
    if load_model == True:
        load_path = "./save/densenet169/mura3cropfocal_1_acc.pkl"

    test_transform = densenet.test_transform
    net = torchvision.models.densenet169(pretrained=not load_model)
    net.classifier = nn.Linear(1664, 1)

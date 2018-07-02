import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MURAdataset import dataset_csv, test_study, get_study_level_data, per_class_dataset
import os
import sys
import numpy as np
import densenet201 as densenet

def sigmoid_test(model, testloader):
    correct = 0
    total = 0
    total_loss = 0
    for data in testloader:
        images, labels, weights = data
        images = images.cuda()
        labels = labels.cuda()
        weights = weights.cuda().float()
        outputs = torch.sigmoid(model(images))
        outputs = outputs.select(1, 0)
        outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
        loss = -(labels.float() * outputs.log() + (1 - labels.float()) * (1 - outputs).log())
        loss = (loss * weights).sum()
        total_loss += float(loss)
        predicted = (outputs >= 0.5).type(torch.cuda.LongTensor)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return (100.0 * float(correct) / float(total)), total_loss / total


study_type = ['XR_WRIST', 'XR_SHOULDER', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS']
pn = ['positive', 'negative']
csv = {}
testset = {}
testloader = {}
acc = list(range(len(study_type)))
count = list(range(len(study_type)))

for i in study_type:
    testset[i] = {}
    testloader[i] = {}

# to be modifed when testing on different model
load_path = './save/densenet169/b8_lr1e-4_d0_logloss.pkl'
root_dir = os.getcwd()
net = torchvision.models.densenet201(pretrained=False)
net.classifier = nn.Linear(1920, 1)
total = 0
result = 0.0

test_transform = densenet.test_transform
print (test_transform)

for i in study_type:
    for j in pn:
        testset[i][j] = per_class_dataset(root_dir, '/traincsv/%s_%s.csv' % (i, j), transform=test_transform, RGB=True)
        total += len(testset[i][j])
        testloader[i][j] = torch.utils.data.DataLoader(testset[i][j], batch_size=8,
                                                       shuffle=False, num_workers=2)

if torch.cuda.device_count() > 1:
    print("Let's use GPUS!")
    # net = nn.DataParallel(net,device_ids=[0,2,3,4,5])

net.load_state_dict(torch.load(load_path))

if torch.cuda.is_available():
    net = net.cuda()

net.eval()

for i in study_type:
    for j in pn:
        acc, loss = sigmoid_test(net, testloader[i][j])
        weight = float(len(testset[i][j])) / float(total)
        result += float(acc) * weight
        print('%s %s %.3f %.3f' % (i, j, acc, loss))

print('final acc: %.3f' % (result))

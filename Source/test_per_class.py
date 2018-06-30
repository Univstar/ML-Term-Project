import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MURAdataset import dataset_csv,test_study,get_study_level_data,per_class_dataset,sigmoid_test, test, model_modify, save_model
import os
import sys
import numpy as np

study_type = ['XR_WRIST','XR_SHOULDER','XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS']
pn = ['positive', 'negative']
csv = {}
testset = {}
testloader = {}
acc = list(range(len(study_type)))
count = list(range(len(study_type)))

for i in study_type:
  testset[i] = {}
  testloader[i] = {}

#to be modifed when testing on different model
load_path = './save/mura3cropfocal_1_loss.pkl'
root_dir = os.getcwd()
net = torchvision.models.densenet169(pretrained=False)
net.classifier = nn.Linear(1664,1)
total = 0
result = 0.0

test_transform=transforms.Compose([transforms.Resize([320,320]),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,                                                         0.224,0.225])        
                                  ])
for i in study_type:
  for j in pn:  
    testset[i][j] = per_class_dataset(root_dir,'/traincsv/%s_%s.csv'%(i,j), transform=test_transform,RGB=True)
    total += len(testset[i][j])
    testloader[i][j] = torch.utils.data.DataLoader(testset[i][j], batch_size=8, 
                                         shuffle=False, num_workers=2)


if torch.cuda.device_count() > 1:
  print("Let's use GPUS!")
  #net = nn.DataParallel(net,device_ids=[0,2,3,4,5])

net.load_state_dict(torch.load(load_path))

if torch.cuda.is_available():
  net = net.cuda()


net.eval()

for i in study_type:
  for j in pn:
    acc,loss = sigmoid_test(net,testloader[i][j]) 
    weight = float(len(testset[i][j]))/float(total)
    result += float(acc) * weight
    print('%s %s %.3f %.3f'%(i,j,acc,loss))

print('final acc: %.3f'%(result))


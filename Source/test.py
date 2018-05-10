import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MURAdataset import dataset_csv,test_study,get_study_level_data,MURAdataset, test, model_modify, save_model
import os
import sys
import numpy as np

study_type = ['XR_WRIST','XR_SHOULDER','XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS']
csv = {}
testset = {}
testloader = {}
acc = list(range(len(study_type)))
count = list(range(len(study_type)))

#to be modifed when testing on different model
load_path = './save/den169_5.pkl'
net = torchvision.models.densenet169(pretrained=False)
net = model_modify(net)


train_transform=transforms.Compose([transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485,), (0.229,)),
                                   ])
test_transform=transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485,), (0.229,)),
                                  ])


for i in study_type:
  csv[i] = get_study_level_data(i,'valid')  

for i in study_type:
  testset[i] = dataset_csv(csv[i], transform=test_transform)
  testloader[i] = torch.utils.data.DataLoader(testset[i], batch_size=1, 
                                         shuffle=False, num_workers=2)


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(),"GPUS!")
  net = nn.DataParallel(net)

net.load_state_dict(torch.load(load_path))

if torch.cuda.is_available():
  net = net.cuda()
    
for i,study in enumerate(study_type,0):
  acc[i], count[i] = test_study(net,testloader[study]) 
  print('study type %s: %.3f'%(study,acc[i]))
 
acc = np.array(acc)
count = np.array(count)
count = count/(count.sum())
final_acc = (acc * count).sum()
print('final accuracy: %.3f'%(final_acc))

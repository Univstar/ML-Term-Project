import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from torch.autograd import Variable
from PIL import Image
import random

study_type = {'XR_WRIST','XR_SHOULDER','XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS'}

class dataset_csv(Dataset):

  def __init__(self, csv_file, transform=None):
    self.csv_list = csv_file
    self.transform = transform

  def __len__(self):
    return len(self.csv_list)

  def __getitem__(self, idx):
    path = self.csv_list.iloc[idx,1]
    count = self.csv_list.iloc[idx,2]
    img_label = self.csv_list.iloc[idx,3]
    imgs = []
 
    for i in range(count):
      img_path = path + 'image%s.png'%(i+1)
      img = Image.open(img_path)
      img = img.convert('L')
      if self.transform:
        img = self.transform(img)
      imgs.append(img)
    
    imgs = torch.stack(imgs)
   
    sample =imgs, img_label, count
    return sample


class MURAdataset(Dataset):

  def __init__(self, root_dir, csv_file, transform=None):
    csv_file_path = root_dir + csv_file
    self.csv_list = pd.read_csv(csv_file_path)
    self.root_dir = root_dir
    self.transform = transform
  
  def __len__(self):
    return len(self.csv_list)

  def __getitem__(self, idx):
    img_name = self.root_dir+'/'+self.csv_list.iloc[idx,0]
    img = Image.open(img_name)
    img = img.convert('L')
    img_label = self.csv_list.iloc[idx,1]
    
    if self.transform:
      img = self.transform(img)

    sample = img, img_label
    return sample

class Imagedataset(Dataset):

    def __init__(self, phase, transform=None):

        self.df = get_whole_study_data(phase)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 1]
        count = self.df.iloc[idx, 2]
        images = get_stacked_img(study_path,count)
        label = self.df.iloc[idx, 3]

        if self.transform:
          images = self.transform(images)
        sample = images, label, count
        return sample


def imagepadding(img):
  
  top = max(0,(224-img.shape[0])//2)
  bottom = max(0,224-img.shape[0]-top)
  left =max(0, (224-img.shape[1])//2)
  right = max(0,(224-img.shape[1])-left)
  img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,0)
  return img

def model_modify(net):
  conv0 = net.features[0]
  tensor = conv0.weight
  tensor = torch.chunk(tensor,3,1)
  tensor = tensor[0]
  newconv = torch.nn.Conv2d(1,64,7,2,3)
  newconv.weight = nn.Parameter(tensor)
  net.features[0] = newconv
  net.classifier = nn.Linear(1664, 2)
  return net

def test(model,testloader):
  correct = 0
  total = 0
  for data in testloader:
      images, labels = data
      outputs = model(Variable(images.cuda())) 
      _, predicted = torch.max(outputs.data, 1)
      total += labels.cuda().size(0)
      correct += (predicted == labels.cuda()).sum()

  return (100 * correct / total)

def test_study(model,testloader):
  correct = 0
  total = 0
  for data in testloader:
    images, labels, count = data
    labels = labels.data.cuda()
    images = images[0]
    outputs = model(Variable(images.cuda()))
    outputs = torch.sum(outputs,0)
    _,predicted = torch.max(outputs,0)
    labels = labels.type(torch.cuda.LongTensor)
    if predicted == labels:
      correct += count
    total += count
  correct = correct.numpy()
  total = total.numpy()
  result = correct[0] / total[0]
  return 100 * result, total[0]


def model_modify3(net):
  net.classifier = nn.Linear(1664, 2)
  return net

def test3(model,testloader):
  correct = 0
  total = 0
  for data in testloader:
      images, labels, count = data
      labels = labels.cuda()
      count = count.cuda()
      outputs = model(Variable(images.cuda()))
      _, predicted = torch.max(outputs.data, 1)
      total += count.sum()
      result = (predicted == labels).float()
      count = count.float()
      correct += (result * count).sum()
  total = total.float()
  return (100 * correct / total)

def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def get_study_level_data(study_type, phase):
    
  study_label = {'positive': 1, 'negative': 0}
  BASE_DIR = 'MURA-v1.0/%s/%s/' % (phase, study_type)
  patients = list(os.walk(BASE_DIR))[0][1]
  study_data = pd.DataFrame(columns=['Type', 'Path', 'Count', 'Label'])
  i = 0
  for patient in patients: # for each patient folder
    for study in os.listdir(BASE_DIR + patient): 
      label = study_label[study.split('_')[1]] # get label 0 or 1
      path = BASE_DIR + patient + '/' + study + '/' # path to this study
      study_data.loc[i] = [study_type, path, len(os.listdir(path)), label] # add new row
      i+=1
  return study_data

def get_whole_study_data(phase):
  whole_study_data = []
  #weights = {} 
  for study in study_type:
    study_data = get_study_level_data(study,phase)
    whole_study_data.append(study_data)
    '''tai = get_count(study_data['train'], 'positive') 
    tni = get_count(study_data['train'], 'negative') 
    Wt1 = n_p(tni / (tni + tai))
    Wt0 = n_p(tai / (tni + tai))
    weights[study] = {'Wt1': Wt1, 'Wt0': Wt0}'''
  whole_study_data = pd.concat(whole_study_data)
  return whole_study_data

def get_stacked_img(path ,count):
  img_list = list(range(0,count))
  while  len(img_list)<3:
    randint = random.randint(0,count-1)
    img_list.append(randint)
  img_list = random.sample(img_list,3)
  imgs = []
  for i in img_list:
    img_path = path + 'image%s.png'%(i+1)
    img = cv2.imread(img_path,0)
    img = cv2.resize(img,(224,224))
    img = img[:,:,np.newaxis]
    imgs.append(img)
  imgs = tuple(imgs)
  res_img = np.concatenate(imgs,2)
  res_img = Image.fromarray(np.uint8(res_img))
  return res_img
         

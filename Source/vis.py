import torch
import torchvision.transforms as transforms
from MURAdataset import MURAdataset
from heatmap import densenet169
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

t = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485,), (0.229,)),
                                  ])
present_dir_path = os.getcwd()
dataset = MURAdataset(csv_file=r'/train.csv', root_dir=present_dir_path,
                                   transform=t)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          shuffle=False, num_workers=0)
weights = torch.load('heat_weight,pkl')
net = densenet169()
net.load_state_dict(weights)

for img in dataloader:
  label = img[1]
  img = img[0]
  heat = net(img)
  predict = heat[0]
  print(predict)
  heat = heat[1]
  _,predict = torch.max(predict,1)
  img = img[0].detach().numpy()
  heat = heat.detach().numpy()
  heat = heat[0]
  heat = np.sum(heat, axis=0)
  heat = heat/heat.max()
  heat = cv2.resize(heat,(224,224),interpolation=cv2.INTER_CUBIC)
  plt.subplot(121)
  print(predict,label[0])
  if predict == label[0]:
    plt.title('correct')
  else:
    plt.title('false')
  plt.imshow(img[0],cmap='gray')
  plt.subplot(122)
  plt.imshow(heat)
  plt.show()

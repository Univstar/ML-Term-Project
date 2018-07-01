import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MURAdataset import MURAdataset
import os
import copy
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image

load_path = "./save/densenet169/mura3cropfocal_1_acc.pkl"
t = transforms.Compose([transforms.Resize((320, 320)),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ])


present_dir_path = os.getcwd()
dataset = MURAdataset(csv_file=r'/train.csv', root_dir=present_dir_path,
                      transform=t)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=0)

net = torchvision.models.densenet169(pretrained=False)
net.classifier = nn.Linear(1664, 1)
net.load_state_dict(torch.load(load_path))

weights = net.classifier.state_dict()["weight"].clone()
print (weights)
weights = weights.view(1, -1, 1, 1)
print (weights.size())


cnt = 0
tot = 0
Max_cnt = 10
for img in dataloader:
    tot += 1
    label = img[1]
    img = img[0]
    predict = torch.sigmoid(net(img))
    if not (label == 1 and predict >= 0.7):
        continue
    print("%d predict = %.3lf" % (tot - 1, predict))

    heat = net.features[:-1](img).clone()
    heat = heat * weights
    heat = torch.mean(heat, dim=1)
    heat = heat.squeeze()

    img = img[0].detach().numpy()

    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406
    img = np.uint8(img * 255)
    img = np.transpose(img, (1, 2, 0))
    cv2.imwrite("./image/image%d.png" % cnt, img)

    heat = heat.detach().numpy()
    heat = cv2.resize(heat, (224, 224), interpolation=cv2.INTER_CUBIC)
    minE = np.amin(heat)
    maxE = np.amax(heat)
    heat = (heat - minE) / (maxE - minE)
    heat = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_HOT)
    cv2.imwrite("./image/image%d_heatmap.png" % cnt, heat)

    alpha, beta = 0.3, 1.0
    combine = cv2.addWeighted(heat, alpha, img, beta, 0)
    cv2.imwrite("./image/image%d_combine.png" % cnt, combine)

    cnt += 1
    if cnt == Max_cnt:
        break

print ("tot = %d, cnt = %d" % (tot, cnt))
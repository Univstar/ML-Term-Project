import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import cv2
import os
import numpy as np
from torch.autograd import Variable
from PIL import Image
import random
from sklearn import metrics

[ 0.82056513,  1.17943487,  1.0036656,   0.9963344,   0.83010581,  1.16989419,
  0.7701037,   1.2298963,   0.74353448,  1.25646552,  0.53545012,  1.46454988,
  0.94744745,  1.05255255]

weights = [ 0.41028257, 0.58971743, 0.5018328,  0.4981672,  0.4150529,  0.5849471,
  0.38505185,  0.61494815, 0.37176724, 0.62823276, 0.26772506,  0.73227494,
  0.47372372,  0.52627628]

study_type = {'XR_WRIST', 'XR_SHOULDER', 'XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS'}
type2num = {'XR_WRIST': 0, 'XR_SHOULDER': 2, 'XR_ELBOW': 4, 'XR_FINGER': 6, 'XR_FOREARM': 8, 'XR_HAND': 10,
            'XR_HUMERUS': 12}


class dataset_csv(Dataset):

    def __init__(self, csv_file, transform=None):
        self.csv_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        path = self.csv_list.iloc[idx, 1]
        count = self.csv_list.iloc[idx, 2]
        img_label = self.csv_list.iloc[idx, 3]
        imgs = []

        for i in range(count):
            img_path = path + 'image%s.png' % (i + 1)
            img = Image.open(img_path)
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)

        sample = imgs, img_label, count
        return sample


def _load_model(model, para):
    total = 0
    new_para = model.state_dict()

    for i in para:
        if i not in new_para:
            print (i)
            total += 1

    model.load_state_dict(para)
    if total == 0:
        print('all parameters loaded')
    else:
        print('%d unloaded parameters' % (total))
    return model


def clip(pred, min):
    for i in range(len(pred)):
        if pred[i] < min:
            pred[i] = min
            print('modify happens')
        if pred[i] > 1 - min:
            pred[i] = 1 - min
            print('modify happens')
    return pred


def load_and_freeze(net, path):
    dic = torch.load(path)
    p = net.state_dict()
    for i in p:
        if not i in dic:
            dic[i] = p[i]
    net.load_state_dict(dic)
    for i, j in net.named_parameters():
        if not 'attention' in i:
            j.requires_grad = False
    return net


class Focal_Loss(torch.nn.modules.Module):
    def __init__(self):
        super(Focal_Loss, self).__init__()

    def forward(self, inputs, targets):
        targets = targets.float()
        inputs = inputs.select(1, 0)
        loss = - (((1 - inputs) ** (1 / 2)) * targets * inputs.log() + (inputs ** (1 / 2)) * (1 - targets) * (
                    1 - inputs).log())
        return loss


class per_class_dataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None, RGB=True):
        csv_file_path = root_dir + csv_file
        self.csv_list = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform
        self.RGB = RGB

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        class_type = self.csv_list.iloc[idx][0]
        class_type = class_type.split('/')[2]
        img_name = self.root_dir + '/' + self.csv_list.iloc[idx][0]
        img = Image.open(img_name)
        if self.RGB == False:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img_label = self.csv_list.iloc[idx, 1]
        weight = weights[type2num[class_type] + img_label]

        if self.transform:
            img = self.transform(img)

        sample = img, img_label, weight
        return sample


class MURAdataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        csv_file_path = root_dir + '/' + csv_file
        self.csv_list = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        img_name = self.root_dir + '/' + self.csv_list.iloc[idx][0]
        img = Image.open(img_name)
        img = img.convert('RGB')
        img_label = self.csv_list.iloc[idx, 1]

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
        images = get_stacked_img(study_path, count)
        label = self.df.iloc[idx, 3]

        if self.transform:
            images = self.transform(images)
        sample = images, label, count
        return sample


def imagepadding(img):
    top = max(0, (224 - img.shape[0]) // 2)
    bottom = max(0, 224 - img.shape[0] - top)
    left = max(0, (224 - img.shape[1]) // 2)
    right = max(0, (224 - img.shape[1]) - left)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    return img


# def model_modify(net, channels1, channels2, Sigmoid=False):
#     conv0 = net.features[0]
#     tensor = conv0.weight
#     tensor = torch.chunk(tensor, 3, 1)
#     tensor = tensor[0]
#     newconv = torch.nn.Conv2d(1, channels1, 7, 2, 3)
#     newconv.weight = nn.Parameter(tensor)
#     net.features[0] = newconv
#     if Sigmoid == False:
#         net.classifier = nn.Linear(channels2, 2)  # 1920 for 201
#     else:
#         net.classifier = nn.Linear(channels2, 1)
#     return net


def test(model, testloader):
    correct = 0
    total = 0
    total_loss = 0
    for data in testloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        total_loss += float(F.cross_entropy(outputs, labels))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.cuda().size(0)
        correct += (predicted == labels.cuda()).sum()

    return (100.0 * correct / total), total_loss


def sigmoid_test(model, testloader):
    correct = 0
    tot_label, tot_predict = np.array([]), np.array([])
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

        tot_label = np.append(tot_label, labels.cpu().numpy())
        tot_predict = np.append(tot_predict, outputs.cpu().detach().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum()

    correct = correct.item()
    # print (tot_label)
    # print (tot_predict)
    score = metrics.roc_auc_score(tot_label, tot_predict)

    return score, 100.0 * correct / total, total_loss / total


# def sigmoid_test(model, testloader):
#     correct = 0
#     total = 0
#     total_loss = 0
#     for data in testloader:
#         images, labels, weights = data
#         images = images.cuda()
#         labels = labels.cuda()
#         weights = weights.cuda().float()
#         outputs = torch.sigmoid(model(images))
#         outputs = outputs.select(1, 0)
#         outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
#         loss = -(labels.float() * outputs.log() + (1 - labels.float()) * (1 - outputs).log())
#         loss = (loss * weights).sum()
#         total_loss += float(loss)
#         predicted = (outputs >= 0.5).type(torch.cuda.LongTensor)
#
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
#
#     return (100.0 * float(correct) / float(total)), total_loss / total

def test_study(model, testloader, Sigmoid=False):
    if Sigmoid == False:
        correct = 0
        total = 0
        for data in testloader:
            images, labels, count = data
            labels = labels.data.cuda()
            images = images[0]
            outputs = model(Variable(images.cuda()))
            outputs = torch.sum(outputs, 0)
            _, predicted = torch.max(outputs, 0)
            labels = labels.type(torch.cuda.LongTensor)
            if predicted == labels:
                correct += count
            total += count
        correct = correct.numpy()
        total = total.numpy()
        result = correct[0] / total[0]
        return 100.0 * result, total[0]
    else:
        correct = 0
        total = 0
        for data in testloader:
            images, labels, count = data
            labels = labels.data.cuda()
            images = images[0]
            outputs = torch.sigmoid(model(Variable(images.cuda())))
            div = count.cuda()
            div = div.type(torch.cuda.FloatTensor)
            outputs = outputs.select(1, 0).sum() / div
            predicted = (outputs > 0.5)
            labels = labels.type(torch.cuda.ByteTensor)
            if predicted == labels:
                correct += count
            total += count
        correct = correct.numpy()
        total = total.numpy()
        result = correct[0] / total[0]
        return 100.0 * result, total[0]


# def model_modify3(net):
#     net.classifier = nn.Linear(1664, 2)
#     return net
#
#
# def test3(model, testloader):
#     correct = 0
#     total = 0
#     for data in testloader:
#         images, labels, count = data
#         labels = labels.cuda()
#         count = count.cuda()
#         outputs = model(Variable(images.cuda()))
#         _, predicted = torch.max(outputs.data, 1)
#         total += count.sum()
#         result = (predicted == labels).float()
#         count = count.float()
#         correct += (result * count).sum()
#     total = total.float()
#     return (100.0 * correct / total)
#
#
# def save_model(model, filename):
#     state = model.state_dict()
#     for key in state: state[key] = state[key].clone().cpu()
#     torch.save(state, filename)


def get_study_level_data(study_type, phase):
    study_label = {'positive': 1, 'negative': 0}
    BASE_DIR = 'MURA-v1.0/%s/%s/' % (phase, study_type)
    patients = list(os.walk(BASE_DIR))[0][1]
    study_data = pd.DataFrame(columns=['Type', 'Path', 'Count', 'Label'])
    i = 0
    for patient in patients:  # for each patient folder
        for study in os.listdir(BASE_DIR + patient):
            label = study_label[study.split('_')[1]]  # get label 0 or 1
            path = BASE_DIR + patient + '/' + study + '/'  # path to this study
            study_data.loc[i] = [study_type, path, len(os.listdir(path)), label]  # add new row
            i += 1
    return study_data


def get_whole_study_data(phase):
    whole_study_data = []
    # weights = {}
    for study in study_type:
        study_data = get_study_level_data(study, phase)
        whole_study_data.append(study_data)
        '''tai = get_count(study_data['train'], 'positive') 
        tni = get_count(study_data['train'], 'negative') 
        Wt1 = n_p(tni / (tni + tai))
        Wt0 = n_p(tai / (tni + tai))
        weights[study] = {'Wt1': Wt1, 'Wt0': Wt0}'''
    whole_study_data = pd.concat(whole_study_data)
    return whole_study_data


def get_stacked_img(path, count):
    img_list = list(range(0, count))
    while len(img_list) < 3:
        randint = random.randint(0, count - 1)
        img_list.append(randint)
    img_list = random.sample(img_list, 3)
    imgs = []
    for i in img_list:
        img_path = path + 'image%s.png' % (i + 1)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (224, 224))
        img = img[:, :, np.newaxis]
        imgs.append(img)
    imgs = tuple(imgs)
    res_img = np.concatenate(imgs, 2)
    res_img = Image.fromarray(np.uint8(res_img))
    return res_img



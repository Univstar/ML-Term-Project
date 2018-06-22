import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MURAdataset import _load_model, clip, Focal_Loss, MURAdataset, per_class_dataset, sigmoid_test, \
    save_model
import os
import sys
import argparse

def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    if v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--net_name', type=str)
    parser.add_argument('--save_path', type=str)
    return parser.parse_args()

def model_modify(net, channels1, channels2):
    conv0 = net.features[0]
    tensor = conv0.weight
    tensor = torch.chunk(tensor, 3, 1)
    tensor = tensor[0]
    newconv = torch.nn.Conv2d(1, channels1, 7, 2, 3)
    newconv.weight = nn.Parameter(tensor)
    net.features[0] = newconv
    net.classifier = nn.Linear(channels2, 1)
    return net

if __name__ == '__main__':
    args = parse_args()
    for k,v in vars(args).items():
        print k, ":", v

    batch_size = args.batch_size
    epoch = args.epoch
    learning_rate = args.learning_rate
    drop_rate = args.drop_rate
    load_model = args.load_model
    if load_model:
        load_path = args.load_path
    net_name = args.net_name
    save_path = args.save_path
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    save_path += net_name

    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485,), (0.229,)),
                                          ])
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485,), (0.229,)),
                                         ])

    present_dir_path = os.getcwd()

    trainset = per_class_dataset(csv_file=r'/train.csv', root_dir=present_dir_path,
                                 transform=train_transform, RGB=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testset = per_class_dataset(csv_file='/valid.csv', root_dir=present_dir_path,
                                transform=test_transform, RGB=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=1)

    net = torchvision.models.densenet169(pretrained=not load_model, drop_rate=drop_rate)
    # net.classifier = nn.Linear(1664,1)
    net = model_modify(net, 64, 1664)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=True)

    if torch.cuda.device_count() > 1:
        print("Let's use %d GPUS!" % torch.cuda.device_count())
        net = nn.DataParallel(net)

    if load_model:
        net = _load_model(net, torch.load(load_path))

    if torch.cuda.is_available():
        net = net.cuda()

    best_acc = 0
    best_model_wts = copy.deepcopy(net.state_dict())

    for step in range(epoch):
        epoch_loss = 0.0
        running_loss = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):

            total += 1

            inputs, labels, weights = data
            weights = weights.type(torch.DoubleTensor)
            labels = labels.type(torch.DoubleTensor)
            inputs, labels, weights = Variable(inputs.cuda()), Variable(labels.cuda()), Variable(weights.cuda())
            optimizer.zero_grad()
            outputs = net(inputs).double()
            outputs = torch.sigmoid(outputs)
            outputs = outputs.select(1, 0)
            outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
            loss = -((1 - outputs) * labels * outputs.log() + outputs * (1 - labels) * (1 - outputs).log())
            loss = (loss * weights).sum()

            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            epoch_loss += loss.data.item()
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' % (step + 1, i + 1, running_loss / 500))
                # sys.stdout.flush()
                running_loss = 0.0

        test_acc, test_loss = sigmoid_test(net, testloader)
        # scheduler.step(test_loss)
        print('test_accuracy and loss in epoch %d : %.3f %.3f' % (step, test_acc, test_loss))
        # sys.stdout.flush()
        if best_acc <= test_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(net.state_dict())

    print ("start to test")
    test_acc, test_loss = sigmoid_test(net, testloader)
    print('test_acc and loss: %.3f %.3f' % (test_acc, test_loss))
    net.load_state_dict(best_model_wts)
    print('Finished Training')
    save_model(net, save_path)
    print('Finished saving')


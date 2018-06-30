import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MURAdataset import _load_model,clip,Focal_Loss, MURAdataset, per_class_dataset, sigmoid_test, model_modify, save_model
import os
import sys

batch_size = 8
epoch = 40
learning_rate = 0.0001
load_model = False
record_result = False
record_path = './result'
load_path = './save/den169_82.pkl'
save_path = './save/mura3cropfocal_1'
net_name = 'densenet169result_4'
if record_result:
  f = open(record_path+'/'+net_name+'.txt','w')
  f.write(net_name)
  f.write('batch_size: %d\nepoch: %d\nlearning_rate: %.5f\n'%(batch_size,epoch,learning_rate))

train_transform = transforms.Compose([transforms.Resize([320,320]),
                                    transforms.CenterCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229                                                         ,0.224,0.225]),
                                   ])
test_transform = transforms.Compose([transforms.Resize((320,320)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,                                                         0.224,0.225])
                                  ])

present_dir_path = os.getcwd() 

trainset = per_class_dataset(csv_file=r'/train.csv', root_dir=present_dir_path,
                                   transform=train_transform,RGB=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=True, num_workers=4)
testset = per_class_dataset(csv_file='/valid.csv', root_dir=present_dir_path,
                                  transform=test_transform,RGB=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, 
                                         shuffle=False, num_workers=1)

net = torchvision.models.densenet169(pretrained=not load_model)
net.classifier = nn.Linear(1664,1)
#net = model_modify(net,64,1664,Sigmoid=True)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=True)

'''if torch.cuda.device_count() > 1:
  print("Let's use GPUS!")
   net = nn.DataParallel(net,device_ids=[0,2,3,4,5])
'''

if load_model:
  net = _load_model(net,torch.load(load_path))

if torch.cuda.is_available():
  net = net.cuda()
 

best_acc = 0
best_loss = 10000 #inf
best_model_wts_acc = copy.deepcopy(net.state_dict())
best_model_wts_loss = copy.deepcopy(net.state_dict())

for step in range(epoch): 
    epoch_loss = 0.0
    running_loss = 0.0
    total = 0
    net.train()
    for i, data in enumerate(trainloader, 0): 
        
        total += 1        

        inputs, labels, weights = data
        weights = weights.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        inputs, labels, weights = Variable(inputs.cuda()), Variable(labels.cuda()), Variable(weights.cuda())  
        optimizer.zero_grad() 
        outputs = net(inputs).double()
        outputs = torch.sigmoid(net(inputs))
        outputs = outputs.select(1,0)
        outputs = torch.clamp(outputs,min=1e-7,max=1-1e-7)
        loss = -( (1-outputs) * labels * outputs.log() +  outputs * (1 - labels) * (1 - outputs).log())
        
        loss = (loss * weights).sum()
        #loss = loss.sum()
        loss.backward() 
        optimizer.step() 
     
        running_loss += loss.data.item()  
        epoch_loss += loss.data.item()
        if i % 500 == 499: 
            print('[%d, %5d] loss: %.3f' % (step+1, i+1, running_loss / 500))
            if record_result: 
                f.write('%.3f\n' %(running_loss / 500))
            sys.stdout.flush()
            running_loss = 0.0
    if step % 1 == 0:
        net.eval()
        test_acc,test_loss = sigmoid_test(net,testloader)
        #scheduler.step(test_loss)
        print('test_accuracy and loss in epoch %d : %.3f %.3f'%(step,test_acc,test_loss))
        print('epoch_loss in epoch %d : %.3f'%(step,epoch_loss/total))     
        if record_result: 
            f.write('test_acc and loss in epoch %d: %.3f %.3f\n'%(step,test_acc,test_loss))
            f.write('epoch_loss in epoch %d: %.3f\n'%(step,epoch_loss/total))
        sys.stdout.flush()
        if best_loss >= test_loss:
            best_loss = test_loss
            best_model_wts_loss = copy.deepcopy(net.state_dict())
        if best_acc <= test_acc:
            best_acc = test_acc
            best_model_wts_acc = copy.deepcopy(net.state_dict())
print('Finished training') 
torch.save(best_model_wts_acc,save_path+'_acc.pkl')
torch.save(best_model_wts_loss,save_path+'_loss.pkl')
print('Finished saving') 


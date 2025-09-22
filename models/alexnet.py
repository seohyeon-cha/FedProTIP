import torch
from torch import nn
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from torch import optim

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()
        track_rs = False
        if args['method'] != 'FedGPM' or args['method'] != 'FOT':
            track_rs = True
        
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.fisher_dict = {}
        self.lamda = 0.1
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=track_rs)
        s=self.compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=track_rs)
        s=self.compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=track_rs)
        s=self.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256*self.smid*self.smid, 1024, bias=False)
        self.bn4 = nn.BatchNorm1d(1024, track_running_stats=track_rs)

        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.bn5 = nn.BatchNorm1d(1024, track_running_stats=track_rs)

        self.map.extend([1024])
        self.out_dim = 1024
        
        self.mapping =  nn.Linear(1024, 1024) # for lander

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        x = x.view(x.shape)
        self.act['conv1.weight'] = x
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.maxpool(self.drop1(self.relu(x)))
        self.act['conv2.weight'] = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv3.weight'] = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool(self.drop2(self.relu(x)))

        x =x.view(bsz,-1)
        self.act['fc1.weight']= x
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.drop2(self.relu(x))
        self.act['fc2.weight'] = x
        self.act['space'] = x
        x = self.fc2(x)
        x = self.bn5(x)
        features = self.drop2(self.relu(x))

        return {
            'features': features,
            'att': self.mapping(features)
        }

    def compute_conv_output_size(self, Lin,kernel_size,stride=1,padding=0,dilation=1):
        return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
    

def get_alexnet(num_task, num_class,args=None):
    return AlexNet(args)
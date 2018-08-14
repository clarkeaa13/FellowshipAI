import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.utils
import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

from PIL import Image, ImageOps

from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import matplotlib.pyplot as plt

import os
import math
import random

###############################################
#Helper Functions
###############################################

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

###############################################
#Build transform
###############################################

transform = T.Compose([
                T.CenterCrop(96),
                T.Resize((32,32),interpolation=1),
                T.ToTensor(),
                T.Normalize((0.924562), (0.264097))
            ])

###############################################
#Build Custom Sampler Class
###############################################

class ClassBalancedSampler(Sampler):
    '''Samples 'num_inst' examples each from 'num_cl' groups. 
    'total_inst' per class, 'total_cl' classes'''

    def __init__(self, num_cl, num_inst, total_cl, total_inst, shuffle=True):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.total_cl = total_cl
        self.total_inst = total_inst
        self.cl_group = np.random.choice(total_cl, num_cl)
        self.ex_group = np.random.choice(total_inst, num_inst)
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items are grouped 20 per class
        batch = []
        for i in self.cl_group:
            for j in self.ex_group:
                batch = batch + [20*i+j]

        if self.shuffle:
            np.random.shuffle(batch)

        return iter(batch)

    # the following functions help you retrieve instances
    # index of original dataset will be 20*class + example
    def get_classes(self):
        return self.cl_group

    def get_examples(self):
        return self.ex_group

    def __len__(self):
        return 1

###############################################
#CUDA stuff
###############################################

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

###############################################
#Build Siamese Networks
###############################################
'''
class CNNEncoder(nn.Module):
    def __init__(self, in_channel, channel_num, out_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channel,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        # output is 4x4

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

class RelationNetwork(nn.Module):
    def __init__(self, channel_num, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(channel_num*2,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

'''
class SiameseNetwork(nn.Module):
    def __init__(self, in_channel, channel_num, hidden_num=channel_num, out_classes=1):
        super().__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channel,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        # output is 4x4 after conv section
        self.fc1 = nn.Sequential(
                    nn.Linear(channel_num*4*4,hidden_num),
                    nn.ReLU())
        # only do last FC layer after difference function
        self.fc2 = nn.Sequential(
                    nn.Linear(hidden_num,out_classes),
                    nn.Sigmoid())
        # input is 1024, output is 64 (channel number)

    def forward_once(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = flatten(out)
        out = self.fc1(out)

    def forward(self,x1,x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out_diff = torch.abs(out1-out2)
        out = self.fc2(out_diff)

###############################################
#Build Dataset
###############################################

print("building dataset...")
omni_train = dset.ImageFolder(root='.\\training_images\\', transform=transform)
#omni_eval = dset.ImageFolder(root='.\\testing_images\\', transform=transform)

###############################################
#Instantiate sampler, DataLoader
###############################################
# make 2 samplers, one for the "sample/training set" of a one-shot classifier
# other sampler is for the "query/test set" which provides many comparisons

print("making samplers...")
train_sample_sampler = ClassBalancedSampler(20,1,963,20)
train_query_sampler = ClassBalancedSampler(20,10,963,20)

print("init DataLoader...")
train_sample_loader = DataLoader(omni_train, batch_size=1, sampler=train_sample_sampler)
train_query_loader = DataLoader(omni_train, batch_size=10, sampler=train_query_sampler) 

# sample datas
samples, sample_labels = train_sample_loader.__iter__().next()
batches, batch_labels = train_query_loader.__iter__().next()
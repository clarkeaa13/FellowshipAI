print("Importing modules...")

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
print("Making helper funcs")

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
#Hyperparameters
###############################################
IMG_SIZE = 32
HIDDEN_LAYER_SIZE = 1*1*64
FC_SIZE = 2*2*64
LEARNING_RATE = 0.001

###############################################
#Function to make optimizer
###############################################

def make_optimizer(NN_Name, LearningRate):
    #input a string representing the name of network
    #learning rate should probably be 0.001 or less
    optimizer = optim.Adam(NN_Name.parameters(), lr = LearningRate)
    return optimizer

###############################################
#Make the target function for MSE loss
###############################################
# Use MSE loss against the relation score output
# Need the target to be 0 for mismatched pair
# and 1 for a matched pair

def make_target(cl1,cl2):
    target = cl1==cl2
    return target

###############################################
#Random weights function (from CS231n)
###############################################
# Could use _calculate_fan_in_and_fan_out if desired
# look up on pytorch docs

def init_weights(m):
    # use by doing $ModuleName$.$layer$.apply(init_weights)
    if type(m) == nn.Conv2d:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif type(m) == nn.Linear:
        n = m.weight.size(1)
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data = torch.ones(m.bias.data.size())

###############################################
#Build transform
###############################################
print("Building Transform")

''' Add random rotations! '''

#degrees = [0,90,180,270]

transform = T.Compose([
                T.Grayscale(),
                T.CenterCrop(104),
                T.Resize((IMG_SIZE,IMG_SIZE),interpolation=1),
                T.ToTensor(),
                T.Normalize([0.924562], [0.264097])
            ])

###############################################
#Build Custom Sampler Classes
###############################################
print("Building Custom Sampler Classes")

class SampleSampler(Sampler):
    '''Samples 'num_inst' examples each from 'num_cl' groups. 
    for one shot learning, num_inst is 1 for sample group.
    'total_inst' per class, 'total_cl' classes'''

    def __init__(self, num_cl=20, total_cl=963, num_inst=1, total_inst=20, shuffle=False):
        self.num_cl = num_cl
        self.total_cl = total_cl
        self.num_inst = num_inst
        self.total_inst = total_inst
        self.cl_list = list(np.random.choice(total_cl, num_cl))
        self.ex_list = list(np.random.randint(total_inst, size=num_inst*20))
        self.shuffle = shuffle
        batch = []
        for i, cl in enumerate(self.cl_list):
            batch = batch + [20*cl+self.ex_list[i]]

        if self.shuffle:
            np.random.shuffle(batch)
        
        self.batch = batch

    def __iter__(self):
        # return a single list of indices, assuming that items are grouped 20 per class
        return iter(self.batch)

    # the following functions help you retrieve instances
    # index of original dataset will be 20*class + example
    def get_classes(self):
        return self.cl_list

    def get_examples(self):
        return self.ex_list

    def get_batch_idc(self):
        return self.batch

    def __len__(self):
        return len(self.batch)

class QuerySampler(Sampler):
    '''Samples queries based on class list and example list'''

    def __init__(self, cl_list, ex_list, num_inst=10, shuffle=True):
        self.cl_list = cl_list
        self.ex_list = ex_list
        self.num_inst = num_inst
        self.shuffle = shuffle
        batch = []
        for i, cl in enumerate(self.cl_list):
            remaining_ex = list(range(20))
            remaining_ex.remove(self.ex_list[i])
            queries = random.sample(remaining_ex, self.num_inst)
            for query in queries:
                batch = batch + [20*cl+query]

        if self.shuffle:
            np.random.shuffle(batch)
        
        self.batch = batch

    def __iter__(self):
        # return a single list of indices, assuming that items are grouped 20 per class
        return iter(self.batch)

    # the following functions help you retrieve instances
    # index of original dataset will be 20*class + example
    def get_classes(self):
        return self.cl_list

    def get_examples(self):
        return self.ex_list

    def get_batch_idc(self):
        return self.batch

    def __len__(self):
        return len(self.batch)

###############################################
#CUDA stuff
###############################################
print("Doing CUDA prep")

USE_GPU = True

dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

###############################################
#Build Siamese Networks
###############################################
print("Building CNNs")

class SiameseNetwork(nn.Module):
    def __init__(self, in_channel=1, channel_num=64, FC_num=1024, hidden_num=256, output_size=1):
        super().__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channel,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(channel_num, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(channel_num, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(channel_num, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(channel_num,channel_num,kernel_size=3,padding=1),
                        nn.BatchNorm2d(channel_num, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        # output is 2x2x64 after conv section

        # input is 2x2x64 = 256, output hidden layer
        # use sigmoid for the FC layers
        self.fc1 = nn.Sequential(
                    nn.Linear(FC_num,hidden_num),
                    nn.Sigmoid())

        # only do last FC layer after difference function
        # input is hidden layer, output is 1 
        self.fc2 = nn.Sequential(
                    nn.Linear(hidden_num,output_size),
                    nn.Sigmoid())

    def forward_once(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = flatten(out)
        out = self.fc1(out)
        return out

    def forward(self,x1,x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out_diff = torch.abs(out1-out2)
        out3 = self.fc2(out_diff)
        return out3

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

###############################################
#Build Dataset
###############################################

#training set has 963 classes 
#test set has has 659 classes
#all images were dumped into two folders for training and testing
#deleted a training class with only 19 images due to an extraction error
print("building datasets...")
omni_train = dset.ImageFolder(root='./training_images/', transform=transform)
omni_test  = dset.ImageFolder(root='./testing_images/', transform=transform)

###############################################
#Function for testing the model
###############################################

def check_accuracy(sample_loader, query_loader, model):
    num_correct = 0
    num_queries = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        model.to(device=device,dtype=dtype)
        for i, (query, query_label) in enumerate(query_loader):
            # for each of 20 queries
            num_queries += 1
            max_score = 0
            truth = None
            for j, (sample, sample_label) in enumerate(sample_loader):
                # check against the 20 one-shot training samples
                sample = sample.to(device=device,dtype=dtype)
                query = query.to(device=device,dtype=dtype)
                score = model(sample, query)
                if query_label==sample_label:
                    truth = j
                if score > max_score:
                    max_score = score
                    hypothesis = j

            if hypothesis == truth:
                num_correct += 1
        acc = num_correct/num_queries
        print('Got %d / %d correct (%.2f)' % (num_correct, num_queries, 100 * acc))

###############################################
#Function to Train the model
###############################################

def train_SN(model, optimizer, episodes=1):
    """Train using siamese network"""
    PATH = './SN_Training/siamese_try'
    PATH2 = './SN_Training/siamese_draft'
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for episode in range(episodes):
        model.train() # set to train mode

        # make the samplers 
        # make 2 samplers, one for the "sample/training set" of a one-shot classifier
        # other sampler is for the "query/test set" which provides many comparisons
        train_sample_sampler = SampleSampler()
        sampled_classes = train_sample_sampler.cl_list
        sampled_examples = train_sample_sampler.ex_list
        train_query_sampler = QuerySampler(sampled_classes, sampled_examples)

        # make the dataloaders
        s_batch_num = 1 # one shot each
        q_batch_num = 10 # pair up 10 per class in a batch (200 total)
        train_sample_loader = DataLoader(omni_train, batch_size=s_batch_num, sampler=train_sample_sampler)
        train_query_loader = DataLoader(omni_train, batch_size=q_batch_num, sampler=train_query_sampler)
        samples, sample_labels = train_sample_loader.__iter__().next()
        #print(samples)
        
        # start training
        scores = torch.zeros(200).to(device=device, dtype=dtype)
        targets = torch.zeros(200).to(device=device, dtype=dtype)
        for i, (sample, sample_label) in enumerate(train_sample_loader):
            for j, (batch, batch_labels) in enumerate(train_query_loader):
                if i==j:
                    targets[i*10:(i+1)*10] = make_target(sample_label, batch_labels)
                    for k in range(10):
                        query = batch[k,:,:,:].to(device=device, dtype=dtype)
                        query = query.view(1,1,IMG_SIZE,IMG_SIZE)
                        sample = sample.to(device=device, dtype=dtype) 
                        scores[k+(i*10)] = model(sample, query)

        # train and update model
        loss = F.mse_loss(scores,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # episodic updates
        if (episode+1)%5 == 0:
            print("episode:",episode+1,"loss",loss.data)

        if (episode+1)%50 == 0:
            ''' Test the model '''
            # make the samplers 
            test_sample_sampler = SampleSampler(total_cl=659)
            sampled_classes = train_sample_sampler.cl_list
            sampled_examples = train_sample_sampler.ex_list
            test_query_sampler = QuerySampler(sampled_classes, sampled_examples, num_inst=1)

            # make the dataloaders
            s_batch_num = 1 # one shot each
            q_batch_num = 1 # one test each
            test_sample_loader = DataLoader(omni_train, batch_size=s_batch_num, sampler=test_sample_sampler)
            test_query_loader = DataLoader(omni_train, batch_size=q_batch_num, sampler=test_query_sampler)
            check_accuracy(test_sample_loader, test_query_loader, model)
            
        if (episode+1)%200 == 0:
            """ Save stuff for training later """
            state = {
                'episode': episode,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, PATH)

        if (episode+1)%500 == 0:
            """ Save as a draft model """
            torch.save(model.state_dict(), PATH)
            

###############################################
#Train the net
###############################################
snet = SiameseNetwork(FC_num=FC_SIZE,hidden_num=HIDDEN_LAYER_SIZE)
snet.layer1.apply(init_weights)
snet.layer2.apply(init_weights)
snet.layer3.apply(init_weights)
snet.layer4.apply(init_weights)
snet.fc1.apply(init_weights)
snet.fc2.apply(init_weights)

snet_optim = make_optimizer(snet,LEARNING_RATE)
print("Begin Training...")
train_SN(snet, snet_optim, episodes = 10000)

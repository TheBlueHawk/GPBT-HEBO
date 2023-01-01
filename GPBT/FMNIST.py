#The 3 following cells aim at simuling the behavior of NN models with a simple model for MNIST
# !pip install ray==1.2.0
# !pip install -U hyperopt
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import random
import math 
from ray import tune
from hyperopt import hp, fmin, tpe, Trials
from functools import *
from ray.tune.logger import *
import copy
import time

EPOCH_SIZE = 32*32*8*32
TEST_SIZE = 256*32*32 #remove 1024

#This is a function that can be used by several NN model
def train(model, optimizer ,func ,train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    model.train()
    #for (data, target) in train_loader:
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > EPOCH_SIZE:
           # print("hehe")
            return
        # We set this just for the example to run quickly.
        data = np.repeat(data, 3, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
       # print(output)
       # print(F.log_softmax(output, dim=1))
       # print(target)
        loss = func(output, target)
        loss.backward()
        optimizer.step()
        
#This is a function that can be used by several NN model (it only does accuracy ATM)
def test(model, func, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            data = np.repeat(data, 3, 1)

            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()


                
    return correct / total


torch.set_num_threads(8)

# A random mnist from the internet to get a correct model to reason about

class train_test_class_fmnist:
    def __init__(self,config):
        self.DEFAULT_PATH = "/tmp/data"
        self.config = {
        "sigmoid_func": 1
      ,  "hidden_dim":64
      ,  "n_layer":3    }
        for key, value in config.items():
            self.config[key] = value
        config = self.config
        
        self.i = 0
        
       # mnist_transforms = transforms.Compose(
       #     [transforms.ToTensor(),
       #      transforms.Normalize((0.1307, ), (0.3081, ))])
        mnist_transforms = transforms.ToTensor()

        self.train_loader = DataLoader(
            datasets.FashionMNIST(self.DEFAULT_PATH, train=True, download=True , transform=mnist_transforms),
            batch_size=1024,
            shuffle=True)
       # self.test_loader = DataLoader(
       #     datasets.MNIST("/gdrive/MyDrive", train=False, transform=mnist_transforms),
       #     batch_size=64,
       #     shuffle=True)

        
        test_valid_dataset = datasets.FashionMNIST(self.DEFAULT_PATH, train=False, transform=mnist_transforms)
        valid_ratio = 0.5  
        nb_test = int((1.0 - valid_ratio) * len(test_valid_dataset))
        nb_valid =  int(valid_ratio * len(test_valid_dataset))
        test_dataset, val_dataset = torch.utils.data.dataset.random_split(test_valid_dataset, [nb_test, nb_valid])
        self.test_loader =  DataLoader(test_dataset,
            batch_size=1024,
            shuffle=True)

        self.val_loader =  DataLoader(val_dataset,
            batch_size=1024,
            shuffle=True)

        sigmoid_func_uniq = nn.Tanh()

        from torchvision import models

        self.model = models.resnet50(num_classes = 30) #LeNet(192,64,10,
                    #3,
                    #config.get("droupout_prob",0.5) ,sigmoid_func_uniq)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get("lr", 0.01),  
                                     amsgrad=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("mps" if torch.backends.mps.is_available() else device)
        self.model.to(device)
    
    def adapt(self, config):
        #print(self.optimizer)
        temp = copy.deepcopy(self)
        for key, value in config.items():
            temp.config[key] = value
        config = temp.config

       # temp.model.adapt(config.get("droupout_prob", 0.5))
        temp.optimizer = torch.optim.Adam(temp.model.parameters(), lr=config.get("lr", 0.01), 
                                     amsgrad=True)
        return temp
    
    # All NN models should have a function train1 and test1 that calls the common train and test defined above.
    # train1 and test1 is then used in the scheduler
    def train1(self):
        print("iteration: " + str(self.i) )
        self.i+=1
        train(self.model, self.optimizer, F.nll_loss, self.train_loader)

    def val1(self):
        return test(self.model, F.nll_loss, self.val_loader)

    def test1(self):
        return test(self.model, F.nll_loss, self.test_loader)

    def step(self):
        self.train1()
        return self.val1()

# __INCEPTION_SCORE_begin__
class LeNet(nn.Module):
    """
    LeNet for MNist classification, used for inception_score
    """

    def __init__(self,input_dim, hidden_dim, output_dim, n_layers,
            drop_prob, sigmoid ):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(drop_prob)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def adapt(self,drop_prob):
        self.conv2_drop = nn.Dropout2d(drop_prob)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



# Convolution Neural network using Pytorch 
class ConvNet(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, n_layers,
                 drop_prob, sigmoid ):
        super(ConvNet, self).__init__()
        
        self.sigmoid = sigmoid
        self.i_d = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)

        self.fc = nn.Linear(input_dim, output_dim)
        self.first= nn.Linear(input_dim, hidden_dim)
        self.hidden = [nn.Linear(hidden_dim,hidden_dim) for _ in range(self.n_layers)]
        self.drop_out = nn.Dropout(drop_prob)

        self.last = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.sigmoid(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.i_d)
        x=self.first(x)
        x=self.drop_out(x)
        for i in range(self.n_layers):
            x=self.hidden[i](x)
            x=self.drop_out(x)
        x = self.last(x)
        return F.log_softmax(x, dim=1)


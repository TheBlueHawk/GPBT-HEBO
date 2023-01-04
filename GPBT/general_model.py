import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from ray import tune
from hyperopt import hp, fmin, tpe, Trials
from functools import *
from ray.tune.logger import *
import copy
from torchvision import models
import time

from nets import LeNet

EPOCH_SIZE = 32 * 32 * 8 * 32
TEST_SIZE = 256 * 32 * 32  # remove 1024

# This is a function that can be used by several NN model
def train(model, optimizer, func, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    model.train()
    # for (data, target) in train_loader:
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > EPOCH_SIZE:
            # print("hehe")
            return
        # We set this just for the example to run quickly.
        # Why this
        # data = np.repeat(data, 3, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        # print(F.log_softmax(output, dim=1))
        # print(target)
        loss = func(output, target)
        loss.backward()
        optimizer.step()


# This is a function that can be used by several NN model (it only does accuracy ATM)
def test(model, func, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            # data = np.repeat(data, 3, 1)

            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


torch.set_num_threads(8)

# A random mnist from the internet to get a correct model to reason about


class general_model:
    def __init__(self, config):
        self.DEFAULT_PATH = "./tmp/data"
        self.config = {"sigmoid_func": 1, "hidden_dim": 64, "n_layer": 3}
        for key, value in config.items():
            self.config[key] = value
        config = self.config
        self.i = 0

        # TODO add MNIST and CIFAR
        mnist_transforms = transforms.ToTensor()
        if config.get("dataset") == "FMNIST":
            self.train_loader = DataLoader(
                datasets.FashionMNIST(
                    self.DEFAULT_PATH,
                    train=True,
                    download=True,
                    transform=mnist_transforms,
                ),
                batch_size=1024,
                shuffle=True,
            )

            test_valid_dataset = datasets.FashionMNIST(
                self.DEFAULT_PATH,
                train=False,
                download=True,
                transform=mnist_transforms,
            )
            valid_ratio = 0.5
            nb_test = int((1.0 - valid_ratio) * len(test_valid_dataset))
            nb_valid = int(valid_ratio * len(test_valid_dataset))
            test_dataset, val_dataset = torch.utils.data.dataset.random_split(
                test_valid_dataset, [nb_test, nb_valid]
            )
            self.test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)
        elif config.get("dataset") == "MNIST":
            self.train_loader = DataLoader(
                datasets.MNIST(
                    self.DEFAULT_PATH,
                    train=True,
                    download=True,
                    transform=mnist_transforms,
                ),
                batch_size=64,
                shuffle=True,
            )
            test_valid_dataset = datasets.MNIST(
                self.DEFAULT_PATH,
                train=False,
                download=True,
                transform=mnist_transforms,
            )
            valid_ratio = 0.5
            nb_test = int((1.0 - valid_ratio) * len(test_valid_dataset))
            nb_valid = int(valid_ratio * len(test_valid_dataset))
            test_dataset, val_dataset = torch.utils.data.dataset.random_split(
                test_valid_dataset, [nb_test, nb_valid]
            )
            self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # TODO add ConvNet and ResNet50
        if config.get("net") == "LeNet":
            self.model = LeNet(
                192, 64, 10, 3, self.config.get("droupout_prob", 0.5), nn.Tanh()
            )

        # mnist_transforms = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.1307, ), (0.3081, ))])

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            betas=_get_betas(config),
            eps=config.get("eps", 1e-08),
            weight_decay=config.get("weight_decay", 0),
            amsgrad=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("mps" if torch.backends.mps.is_available() else device)
        self.model.to(device)

    def adapt(self, config):
        temp = copy.deepcopy(self)
        for key, value in config.items():
            temp.config[key] = value
        config = temp.config

        temp.model.adapt(config.get("droupout_prob", 0.5))
        temp.optimizer = torch.optim.Adam(
            temp.model.parameters(),
            lr=config.get("lr", 0.01),
            betas=_get_betas(config),
            eps=config.get("eps", 1e-08),
            weight_decay=config.get("weight_decay", 0),
            amsgrad=True,
        )
        return temp

    # All NN models should have a function train1 and test1 that calls the common train and test defined above.
    # train1 and test1 is then used in the scheduler
    def train1(self):
        print("iteration: " + str(self.i))
        self.i += 1
        train(self.model, self.optimizer, F.nll_loss, self.train_loader)

    def val1(self):
        return test(self.model, F.nll_loss, self.val_loader)

    def test1(self):
        return test(self.model, F.nll_loss, self.test_loader)

    def step(self):
        self.train1()
        return self.val1()


def _get_betas(config):
    if config.get("b1", 0.999) >= 1:
        b1 = 1 - 1e-10
    else:
        b1 = 1 - config.get("b1", 0.999)

    if config.get("b2", 0.999) >= 1:
        b2 = 1 - 1e-10
    else:
        b2 = 1 - config.get("b2", 0.999)
    return (b1, b2)

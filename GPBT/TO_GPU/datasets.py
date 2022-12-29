import copy
#from torchvision import datasets as tv_datasets
#from torchvision import transforms as tv_transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

#import pandas as pd
#from sklearn import datasets as skl_datasets
#from sklearn.model_selection import train_test_split
# custom imports from the same folder
import model


# The following two functions are general functions used for all different choice of
# optimizers, models, schedulers, ...

EPOCH_SIZE = 60*1000  # 10 ierations to train the whole dataset once
TEST_SIZE = 10*1000

# This is a function that can be used by several NN model
# the device allocation is done before calling it in the function it only checks
# if the devices are the same


def train(model, optimizer, loss_func, train_loader):
    # should add the check that all three devices are the same
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            break
        # We set this just for the example to run quickly.
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

# This is a function that can be used by several NN model (it only does accuracy ATM)


def test(model, data_loader, type_of_problem):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            if(type_of_problem == 1):  # classification
                # We set this just for the example to run quickly.
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            if(type_of_problem == 2):  # regression
                data, target = data.to(device), target.to(device)
                # ??? there is no more need for the Variable since we are not using autograd
                from torch.autograd import Variable

                X = Variable(torch.FloatTensor(data))
                result = model(X)
                pred = result.data[:, 0].numpy()
                out = target.data[:, 0].numpy()

                total += target.size(0)

                correct += torch.nn.MSELoss()(result, target).numpy()

    return correct / total


class TrainMnist():
    """Used to train different models from `models.py` on the MNIST dataset
    """

    aviable_models = {
        "LeNet": "It is a Convolutional Neural Network.",
        "ConvNet": "It is a ...",
        "LinearReg": "It is a ..."
    }
    
    # dafault parameters values
    config = {
        "sigmoid_func": 1,
        "hidden_dim": 43,
        "n_layer": 2
    }

    valid_ratio = 0.5

    def __init__(self, config):

        for key, value in config.items():
            self.config[key] = value

        config = self.config
        self.i = 0

        mnist_transforms = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Train Set Initialiazation
        self.train_loader = DataLoader(
            tv_datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
            batch_size=config.get("batch_size", 64),
            shuffle=True
        )

        # Validation and Test Set Initialization
        test_valid_dataset = tv_datasets.MNIST("~/data", train=False, transform=mnist_transforms)

        nb_test = int((1.0 - self.valid_ratio) * len(test_valid_dataset))
        nb_valid = int(self.valid_ratio * len(test_valid_dataset))
        test_dataset, val_dataset = torch.utils.data.dataset.random_split(test_valid_dataset, [nb_test, nb_valid])

        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # Initialization of the model
        if True:
            self.model = model.LeNet(
                192,
                int(round(config.get("hidden_dim", 64))),
                10,
                int(round(config.get("n_layer", 1))),
                config.get("droupout_prob", 0.5),
                nn.Tanh()
            )
        elif True:
            raise NotImplementedError
        else:
            raise ValueError("Model with name {} is not recognized.".format(model_name))

        # Method of Optimization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            betas=((config.get("b1", 0.999), config.get("b2", 0.9999))),
            eps=config.get("eps", 1e-08),
            weight_decay=config.get("weight_decay", 0),
            amsgrad=True
        )

    def get_aviable_models(self, show=False):
        """Returns the supported models with a brief description

        Args:
            show (bool, optional): Decide if the models should be printed to the console. Defaults to False.

        Returns:
            dict: Dictionary with name of the aviable models and a description.
        """
        if show:
            for key, value in self.aviable_models.items():
                print("Key: {}\tDescription: {}".format(key, value))

        return self.aviable_models

    def adapt(self, config):
        """[summary]

        Args:
            config ([type]): [description]

        Returns:
            [type]: [description]
        """
        # what is it doing
        self_copy = copy.deepcopy(self)
        for key, value in config.items():
            self_copy.config[key] = value

        # propagate the adapt call to the actual model
        config = self_copy.config
        self_copy.model.adapt(config.get("droupout_prob", 0.5))

        self_copy.optimizer = torch.optim.Adam(
            self_copy.model.parameters(),
            lr=config.get("lr", 0.01),
            betas=((config.get("b1", 0.999), config.get("b2", 0.9999))),
            eps=config.get("eps", 1e-08),
            weight_decay=config.get("weight_decay", 0),
            amsgrad=True
        )
        return self_copy

    def train1(self):
        # print("iteration: {}".format(self.i))
        self.i += 1
        train(self.model, self.optimizer, F.nll_loss, self.train_loader)

    def val1(self):
        return test(self.model, self.val_loader, 1)

    def test1(self):
        return test(self.model, self.test_loader, 1)

    # what train1 is it calling ? is it
    def step(self):
        self.train1()
        return self.val1()


class TrainBoston():
    """Used to train different models from `models.py` on the Boston Housing dataset as a Regression Problem.
    """

    aviable_models = {
        "NeurNet": "It is a Convolutional Neural Network."
    }

    # here we should insert the default parameters
    config = {

    }

    valid_ratio = 0.5
    test_ratio = 0.2

    def __init__(self, config):
        data = skl_datasets.load_boston()

        # importing the data into pandas dataframe
        X = pd.DataFrame(data.data, columns=data.feature_names)
        Y = pd.DataFrame(data.target, columns=["MEDV"])

        # normalizatoion of features
        Y = Y.apply(lambda x: (x - x.mean()) / x.std())

        #
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_ratio, random_state=1234)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=self.valid_ratio, random_state=1234)

        split_data = [X_train, y_train, X_test, y_test, X_val, y_val]
        x_train, y_train, x_test, y_test, x_val, y_val = [torch.tensor(x.values, dtype=torch.float) for x in split_data]

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=100,
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test),
            batch_size=100,
            shuffle=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_val, y_val),
            batch_size=100,
            shuffle=True
        )

        for key, value in config.items():
            self.config[key] = value

        config = self.config
        self.i = 0

        # Initialization of the model
        if True:
            self.model = model.NeurNet(
                13,
                int(round(self.config.get("hidden_dim", 64))),
                1,
                int(round(self.config.get("n_layer", 1))),
                self.config.get("droupout_prob", 0.1),
                F.relu
            )
        elif True:
            raise NotImplementedError
        else:
            raise ValueError(
                "Model with name {} is not recognized.".format(model_name))

        # Method of Optimization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("lr"),
            betas=((self.config.get("b1"), self.config.get("b2"))),
            eps=10.0 ** - (self.config.get("eps")),
            weight_decay=self.config.get("weight_decay"),
            amsgrad=True
        )

    def get_aviable_models(self, show=False):
        """Returns the supported models with a brief description

        Args:
            show (bool, optional): Decide if the models should be printed to the console. Defaults to False.

        Returns:
            dict: Dictionary with name of the aviable models and a description.
        """
        if show:
            for key, value in self.aviable_models.items():
                print("Key: {}\tDescription: {}".format(key, value))

        return self.aviable_models

    def adapt(self, config):
        """[summary]

        Args:
            config ([type]): [description]

        Returns:
            [type]: [description]
        """
        # what is it doing
        self_copy = copy.deepcopy(self)
        for key, value in config.items():
            self_copy.config[key] = value

        # propagate the adapt call to the actual model
        config = self_copy.config
        self_copy.model.adapt(config.get("droupout_prob", 0.5))

        self_copy.optimizer = torch.optim.Adam(
            self_copy.model.parameters(),
            lr=config.get("lr"),
            betas=((config.get("b1"), config.get("b2"))),
            eps=10.0**-config.get("eps"),
            weight_decay=config.get("weight_decay"),
            amsgrad=True
        )
        return self_copy

    def train1(self):
        #print("iteration: {}".format(self.i))
        self.i += 1
        train(self.model, self.optimizer, nn.MSELoss(), self.train_loader)

    def val1(self):
        return -test(self.model, self.val_loader, 2)

    def test1(self):
        return -test(self.model, self.test_loader, 2)

    # what train1 is it calling ? is it
    def step(self):
        self.train1()
        return self.val1()


class TrainCIFAR():
    """Used to train different models from `models.py` on the CIFAR dataset
    """
    aviable_models = {
        "AlexNet": "It is a ...",
        "ConvNet": "It is a ...",
        "LinearReg": "It is a ..."
    }

    # here we should insert the default parameters
    config = {

    }

    valid_ratio = 0.5

    def __init__(self, config):

        cifar_transf = tv_transforms.Compose([tv_transforms.ToTensor()])

        # Train Set Initialiazation from torch vision
        self.train_loader = DataLoader(
            tv_datasets.CIFAR10("~/data", train=True, download=True, transform=cifar_transf),
            batch_size=64,
            shuffle=True
        )

        # Validation and Test Set Initialization
        # why not put it in a dataloader ?
        test_valid_dataset = tv_datasets.CIFAR10("~/data", train=False, download=True, transform=cifar_transf)

        
        nb_test = int((1.0 - self.valid_ratio) * len(test_valid_dataset))
        nb_valid = int(self.valid_ratio * len(test_valid_dataset))
        test_dataset, val_dataset = torch.utils.data.dataset.random_split(test_valid_dataset, [nb_test, nb_valid])

        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        for key, value in config.items():
            self.config[key] = value

        config = self.config
        self.i = 0

        # Initialization of the model
        # BIG mistake
        if True:
            # many of the parameters are not even used in init
            self.model = models.AlexNet(config.get("droupout_prob"))
        elif True:
            raise NotImplementedError
        else:
            raise ValueError(
                "Model with name {} is not recognized.".format(model_name))

        # Method of Optimization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("lr"),
            betas=((self.config.get("b1"), config.get("b2"))),
            eps=10.0 ** (-self.config.get("eps")),
            weight_decay=self.config.get("weight_decay"),
            amsgrad=True
        )

    def get_aviable_models(self, show=False):
        """Returns the supported models with a brief description

        Args:
            show (bool, optional): Decide if the models should be printed to the console. Defaults to False.

        Returns:
            dict: Dictionary with name of the aviable models and a description.
        """
        if show:
            for key, value in self.aviable_models.items():
                print("Key: {}\tDescription: {}".format(key, value))

        return self.aviable_models

    def adapt(self, config):
        """[summary]

        Args:
            config ([type]): [description]

        Returns:
            [type]: [description]
        """
        # what is it doing
        self_copy = copy.deepcopy(self)
        for key, value in config.items():
            self_copy.config[key] = value

        # propagate the adapt call to the actual model
        config = self_copy.config
        self_copy.model.adapt(config.get("droupout_prob", 0.5))

        self_copy.optimizer = torch.optim.Adam(
            self_copy.model.parameters(),
            lr=config.get("lr"),
            betas=((config.get("b1"), config.get("b2"))),
            eps=10.0**-config.get("eps"),
            weight_decay=config.get("weight_decay"),
            amsgrad=True
        )

        # why need to return the copy of itself?
        return self_copy

    def train1(self, train_loader):
     #   print("iteration: {}".format(self.i))
        self.i += 1
        train(self.model, self.optimizer, F.nll_loss, train_loader)

    def val1(self, val_loader):
        return test(self.model, val_loader, 1)

    def test1(self, test_loader):
        return test(self.model, test_loader, 1)

    # what train1 is it calling ? is it
    def step(self):
        self.train1()
        return self.val1()

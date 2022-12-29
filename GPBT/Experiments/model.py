import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py

NUM_CLASSES = 10


class AlexNet(nn.Module):
    """ AlexNet Convolutional Neural Network porposed by Alex Krizhevsky et al. in 2012
    """
    def __init__(self, prob):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, NUM_CLASSES),
        )
        self.conv1_drop = nn.Dropout2d(prob)
        self.conv2_drop = nn.Dropout2d(prob)

    def adapt(self, drop_prob):
        self.conv1_drop = nn.Dropout2d(drop_prob)
        self.conv2_drop = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.conv1_drop(x)
        x = self.classifier1(x)
        x = self.conv2_drop(x)
        x = self.classifier2(x)
        return x


class NeurNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
                 drop_prob, sigmoid):
        super(NeurNet, self).__init__()

        self.sigmoid = sigmoid
        self.i_d = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.first = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.drop_out = nn.Dropout(drop_prob)

        self.last = nn.Linear(hidden_dim, output_dim)

    def adapt(self, drop_prob):
        self.drop_out = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = x.view(-1, self.i_d)
        x = self.first(x)
        x = self.drop_out(x)
        for _ in range(self.n_layers):
            x = self.hidden(x)
            x = self.drop_out(x)
        x = self.last(x)
        return x


class LeNet(nn.Module):
    """LeNet Convolutional Neural Network proposed by Yann LeCun et al. in 1989
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob, sigmoid):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(drop_prob)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def adapt(self, drop_prob):
        self.conv2_drop = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob, sigmoid):
        super(ConvNet, self).__init__()

        self.sigmoid = sigmoid
        self.i_d = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)

        self.fc = nn.Linear(input_dim, output_dim)
        self.first = nn.Linear(input_dim, hidden_dim)
        self.hidden = [nn.Linear(hidden_dim, hidden_dim)
                       for _ in range(self.n_layers)]
        self.drop_out = nn.Dropout(drop_prob)

        self.last = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.sigmoid(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, self.i_d)
        x = self.first(x)
        x = self.drop_out(x)

        for i in range(self.n_layers):
            x = self.hidden[i](x)
            x = self.drop_out(x)

        x = self.last(x)
        return F.log_softmax(x, dim=1)


class LinearReg(nn.Module):
    def __init__(self, config):
        super(LinearReg, self).__init__()

        hidden_dims = config.get("hidden_dims", [150, 100, 75])
        self.linears = nn.ModuleList(
            [nn.Linear(28*28, hidden_dims[0], bias=True), nn.ReLU()])

        for i in range(1, len(hidden_dims)):
            self.linears.append(
                nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=True))
            self.linears.append(nn.ReLU())

        # ???
        self.model = Net(self.linears)

    def forward(self, x):
        for _, layer in enumerate(self.linears):
            x = layer(x)
        return x


class Net(nn.Module):
    def __init__(self, linears):
        super(Net, self).__init__()
        self.linears = linears

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = layer(x)
        return x


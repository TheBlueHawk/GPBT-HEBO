import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet for MNist classification, used for inception_score
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
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Convolution Neural network using Pytorch
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
        self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_layers)]
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

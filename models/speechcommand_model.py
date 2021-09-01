import torch.nn as nn
import torch.nn.functional as F

from .base_models import *


class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16280, 1000)
        self.fc2 = nn.Linear(1000, 30)

    def forward(self, x, feat=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        if feat: return x
        x = self.fc2(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_size=100, large=False):
        super(Generator, self).__init__()

        self.noise_size = noise_size

        self.core_net = nn.Sequential(
            nn.Linear(self.noise_size, 200 * 8 * 5, bias=False), nn.BatchNorm1d(200 * 8 * 5), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0), 200, 8, 5)),
            nn.ConvTranspose2d(200, 96, (2, 2), (2, 2), (0, 0), (0, 0), bias=False), nn.BatchNorm2d(96), nn.LeakyReLU(),
            nn.ConvTranspose2d(96, 64, (2, 2), (2, 2), (0, 0), (0, 0), bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            WN_ConvTranspose2d(64, 1, (5, 5), (5, 5), (0, 0), (1, 1), train_scale=True, init_stdv=0.1), nn.Tanh(),
            # Expression(lambda tensor: tensor.view(tensor.size(0), 1, 161, 101))
        )

    def forward(self, noise):
        out = self.core_net(noise)

        return out


class Discriminative(nn.Module):
    def __init__(self):
        super(Discriminative, self).__init__()

        self.num_label = 10

        n_filter_1, n_filter_2, n_filter_3 = 64, 96, 200

        # Assume X is of size [batch x 3 x 32 x 32]
        self.core_net = nn.Sequential(

            nn.Sequential(nn.Dropout(0.15)),
            WN_Conv2d(1, n_filter_1, 5, 5, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 1, 1, 0), nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            WN_Conv2d(n_filter_1, n_filter_2, 2, 2, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            WN_Conv2d(n_filter_2, n_filter_3, 2, 2, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_3, n_filter_3, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_3, n_filter_3, 1, 1, 0), nn.LeakyReLU(0.2),
            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        self.out_net = WN_Linear(n_filter_3, self.num_label, train_scale=True, init_stdv=0.1)

    def forward(self, x, feat=False):
        if feat:
            return self.core_net(x)
        else:
            return self.out_net(self.core_net(x))


def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 30)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out)


class Distribution_Matching_Network(nn.Module):
    def __init__(self, input_size=128, dataset="other"):
        super(Distribution_Matching_Network, self).__init__()

        if dataset == "cifar10":
            input_size = 192

        if dataset == "svhn":
            self.core_net = nn.Sequential(
                nn.Linear(input_size, input_size), nn.ReLU(),
                nn.Linear(input_size, 2)
            )
        elif dataset == "cifar10":
            self.core_net = nn.Sequential(
                nn.Linear(input_size, input_size), nn.ReLU(),
                nn.Linear(input_size, input_size), nn.ReLU(),
                nn.Linear(input_size, 2)
            )

        else:
            input_size = 200
            self.core_net = nn.Sequential(
                nn.Linear(input_size, input_size), nn.ReLU(),
                nn.Linear(input_size, input_size), nn.ReLU(),
                nn.Linear(input_size, 2)
            )

    def forward(self, input):
        output = self.core_net(input)

        return output

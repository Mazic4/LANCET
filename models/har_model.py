import numpy as np
import random
import time
import os, sys
import math
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10
import torchvision


from .base_models import *


class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        self.conv1 = nn.Sequential(

            WN_Conv2d(in_channels=9, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),

            nn.LeakyReLU(),

            nn.Dropout(0.2),

        )

        self.conv2 = nn.Sequential(

            WN_Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),

            nn.LeakyReLU(),

            nn.Dropout(0.2),

        )

        self.conv3 = nn.Sequential(

            WN_Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 2), stride=(1, 2)),

            nn.LeakyReLU(),

            nn.Dropout(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2)),
        )

        self.fc1 = nn.Sequential(

            WN_Linear(64, 6, train_scale=True, init_stdv=0.1)

        )

    def forward(self, x, feat=False):

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if feat == True:

            return out

        else:

            out = self.fc1(out)

            return out


class Expression(nn.Module):

    def __init__(self, func):
        super(Expression, self).__init__()

        self.func = func

    def forward(self, input):
        return self.func(input)


class Generator(nn.Module):

    def __init__(self, image_size=32, noise_size=100, large=False):
        super(Generator, self).__init__()

        self.noise_size = noise_size

        self.image_size = image_size

        self.core_net = nn.Sequential(

            nn.Linear(self.noise_size, 128 * 16, bias=False), nn.BatchNorm1d(128 * 16), nn.ReLU(),

            Expression(lambda tensor: tensor.view(tensor.size(0), 128, 1, 16)),

            nn.ConvTranspose2d(128, 64, (1, 2), (1, 2), (0, 0), (0, 0), bias=False), nn.BatchNorm2d(64), nn.ReLU(),

            nn.ConvTranspose2d(64, 32, (1, 2), (1, 2), (0, 0), (0, 0), bias=False), nn.BatchNorm2d(32), nn.ReLU(),

            WN_ConvTranspose2d(32, 9, (1, 2), (1, 2), (0, 0), (0, 0), train_scale=True, init_stdv=0.1), nn.Tanh(),

            Expression(lambda tensor: tensor.view(tensor.size(0), 9, 1, 128))

        )

    def forward(self, noise):
        out = self.core_net(noise)

        return out


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)


class Discriminator2(nn.Module):

    def __init__(self, feature_size, large=False):
        super(Discriminator2, self).__init__()

        self.core_net = nn.Sequential(

            nn.Linear(feature_size, 100, bias=False), nn.ReLU(),

            nn.Linear(100, 100, bias=False), nn.Dropout(0.1), nn.ReLU(),

            nn.Linear(100, 100, bias=False), nn.Dropout(0.1), nn.ReLU(),

            nn.Linear(100, 100, bias=False), nn.Dropout(0.1), nn.ReLU(),

            nn.Linear(100, 2, bias=False), nn.Tanh(),

            Expression(lambda tensor: tensor.view(tensor.size(0), 2))

        )

    def forward(self, feature):
        out = self.core_net(feature)

        return out

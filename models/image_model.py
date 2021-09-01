import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


from .base_models import *





class Discriminative(nn.Module):
    def __init__(self, config):
        super(Discriminative, self).__init__()

        print ('===> Init small-conv for {}'.format(config.dataset))

        #self.noise_size = config.noise_size
        self.num_label  = config.num_label

        if config.dataset == 'svhn':
            n_filter_1, n_filter_2 = 64, 128
        elif config.dataset == 'cifar':
            n_filter_1, n_filter_2 = 96, 192
        else:
            raise ValueError('dataset not found: {}'.format(config.dataset))

        # Assume X is of size [batch x 3 x 32 x 32]
        self.core_net = nn.Sequential(

            nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) if config.dataset == 'svhn' \
                else nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.2)),

            WN_Conv2d(         3, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5) if config.dataset == 'svhn' else nn.Dropout(0.5),

            WN_Conv2d(n_filter_1, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5) if config.dataset == 'svhn' else nn.Dropout(0.5),

            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        self.out_net = WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)

    def forward(self, X, feat=False):
        if X.dim() == 2:
            X = X.view(X.size(0), 3, 32, 32)
        
        if feat:
            return self.core_net(X)
        else:
            return self.out_net(self.core_net(X))

class Generator(nn.Module):
    def __init__(self, image_size, noise_size=100, large=False):
        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        if not large:
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 4 * 4 * 512, bias=False), nn.BatchNorm1d(4 * 4 * 512), nn.ReLU(),
                Expression(lambda tensor: tensor.view(tensor.size(0), 512, 4, 4)),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128,   3, 5, 2, 2, 1, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )
        else:
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 2 * 2 * 1024, bias=False), nn.BatchNorm1d(2 * 2 * 1024), nn.ReLU(),
                Expression(lambda tensor: tensor.view(tensor.size(0), 1024, 2, 2)),
                nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128,   3, 5, 1, 2, 0, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )

    def forward(self, noise):
        output = self.core_net(noise)

        return output

class Encoder(nn.Module):
    def __init__(self, image_size, noise_size=100, output_params=False):
        super(Encoder, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        self.core_net = nn.Sequential(
            nn.Conv2d(  3, 128, 5, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2, 2, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0), 512 * 4 * 4)),
        )
        
        if output_params:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(4 * 4 * 512, self.noise_size*2, train_scale=True, init_stdv=0.1))
            self.core_net.add_module(str(len(self.core_net._modules)), Expression(lambda x: torch.chunk(x, 2, 1)))
        else:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(4 * 4 * 512, self.noise_size, train_scale=True, init_stdv=0.1))

    def forward(self, input):
        
        output = self.core_net(input)

        return output


class Distribution_Matching_Network(nn.Module):
    def __init__(self, input_size = 128, dataset = "svhn"):
        super(Distribution_Matching_Network, self).__init__()
        
        if dataset=="cifar10":
            input_size = 192

        elif dataset=="svhn":
            input_size = 128
        
        if dataset == "svhn":
            self.core_net = nn.Sequential(
                nn.Linear(input_size, input_size),nn.ReLU(),
                nn.Linear(input_size, 2)
            )
        elif dataset == "cifar10":
            self.core_net = nn.Sequential(
                nn.Linear(input_size, input_size),nn.ReLU(),
                nn.Linear(input_size, input_size),nn.ReLU(),
                nn.Linear(input_size, 2)
            )

        else:
            input_size = 192
            self.core_net = nn.Sequential(
                nn.Linear(input_size, input_size),nn.ReLU(),
                nn.Linear(input_size, 2)
            )

    def forward(self, input):
        output = self.core_net(input)

        return output


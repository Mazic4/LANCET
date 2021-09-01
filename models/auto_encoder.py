import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter



#This autoencoder model is modified based on:
# https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py

class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        
        if config.dataset in ["cifar", "svhn"]:
            self.input_channels = 3
            kernal_size_1, kernal_size_2 = 4,4
            stride_size_1, stride_size_2 = 2,2
            padding_size = 1
            output_padding_size = 0
        elif config.dataset == "har":
            self.input_channels = 9
            kernal_size_1,kernal_size_2 = (1,2), (1,2)
            stride_size_1, stride_size_2 = (1,2), (1,2)
            padding_size = 0
            output_padding_size = 0
        elif config.dataset == "speechcommand":
            self.input_channels = 1
            kernal_size_1 = 5
            kernal_size_2 = 2
            stride_size_1, stride_size_2 = 5, 2
            padding_size = 0
            #the output_padding size for the last layer
            output_padding_size = 1
        else:
            raise ValueError("Unkown Dataset: {}".format(config.dataset))
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 12, kernal_size_1, stride=stride_size_1, padding=padding_size),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, kernal_size_2, stride=stride_size_2, padding=padding_size),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, kernal_size_2, stride=stride_size_2, padding=padding_size),           # [batch, 48, 4, 4]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, kernal_size_2, stride=stride_size_2, padding=padding_size),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, kernal_size_2, stride=stride_size_2, padding=padding_size),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, self.input_channels, kernal_size_1, stride=stride_size_1, padding=padding_size, output_padding = output_padding_size),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded










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


def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')

        # Read dataset from disk, dealing with text files' syntax

        X_signals.append(

            [np.array(serie, dtype=np.float32) for serie in [

                row.replace('  ', ' ').strip().split(' ') for row in file

            ]]

        )

        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
        file = open(y_path, 'r')

        # Read dataset from disk, dealing with text file's syntax

        y_ = np.array(

            [elem for elem in [

                row.replace('  ', ' ').strip().split(' ') for row in file

            ]],

            dtype=np.int32

        )

        file.close()

        # Substract 1 to each output class for friendly 0-based indexing

        return y_ - 1


def create_har_dataset():
    
    INPUT_SIGNAL_TYPES = [
    
        "body_acc_x_",
    
        "body_acc_y_",
    
        "body_acc_z_",
    
        "body_gyro_x_",
    
        "body_gyro_y_",
    
        "body_gyro_z_",
    
        "total_acc_x_",
    
        "total_acc_y_",
    
        "total_acc_z_"
    
    ]
    
    LABELS = [
    
        "WALKING",
    
        "WALKING_UPSTAIRS",
    
        "WALKING_DOWNSTAIRS",
    
        "SITTING",
    
        "STANDING",
    
        "LAYING"
    
    ]
    
    DATASET_PATH = "/home/zhanghuayi01/lancet/LANCET/data/UCI_HAR_Dataset/"
    
    TRAIN = "train/"
    
    TEST = "test/"
    
    X_train_signals_paths = [
    
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    
    ]
    
    X_test_signals_paths = [
    
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    
    ]
    
    X_train = load_X(X_train_signals_paths)
    
    X_test = load_X(X_test_signals_paths)
    
    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    
    y_train = load_y(y_train_path)
    
    y_test = load_y(y_test_path)

    feats = []

    np.random.seed(1)

    index = np.arange(len(X_train))

    np.random.shuffle(index)

    X_train = X_train[index]

    y_train = y_train[index]

    raw_dataset = []
    for i in range(len(X_train)):
        image = X_train[i].T[:, np.newaxis, :]
        image /= np.max(image)
        raw_dataset += [(torch.from_numpy(image), y_train[i])]

    raw_dataset = np.array(raw_dataset)

    test_dataset = []
    for i in range(len(X_test)):
        image = X_test[i].T[:, np.newaxis, :]
        image /= np.max(image)
        test_dataset += [(torch.from_numpy(image), y_test[i])]

    test_dataset = np.array(test_dataset)

    return raw_dataset, test_dataset

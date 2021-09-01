import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100


class DataLoader(object):

    def __init__(self, config, raw_loader, indices, batch_size, labels = 'none'):
        self.images, self.labels = [], []
        for i, idx in enumerate(indices):
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)
    
        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()

        self.batch_size = batch_size
        self.indices = indices

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)
        self.propagate_indices = np.array([])

        print ("The len of data loader is", len(self.images))


    def get_zca_cuda(self, reg=1e-6):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        mean = images.mean(0)
        images -= mean.expand_as(images)
        sigma = torch.mm(images.transpose(0, 1), images) / images.size(0)
        U, S, V = torch.svd(sigma)
        components = torch.mm(torch.mm(U, torch.diag(1.0 / torch.sqrt(S) + reg)), U.transpose(0, 1))
        return components, mean

    def apply_zca_cuda(self, components):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        self.images = torch.mm(images, components.transpose(0, 1)).cpu()

    def generator(self, inf=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices).long()
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels, self.indices[indices[start:end]]
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len

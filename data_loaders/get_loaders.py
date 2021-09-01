import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100


from .dataloader import DataLoader
from .gcommand_loader import GCommandLoader
from .har_make_dataset import create_har_dataset


def get_mnist_loaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = MNIST(config.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: config.size_labeled_data // 10]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0])

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels == i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set


def get_svhn_loaders(config, load_indices=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(config.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(config.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        for i in range(len(data_set.data)):
            if data_set.labels[i] == 10:
                data_set.labels[i] = 0

    preprocess(training_set)
    preprocess(dev_set)

    np.random.seed(1)

    indices = np.arange(len(training_set))
    # np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)

    intermidiate_path = "{}svhn_labeled_indices_{}.txt".format(config.save_dir, config.size_labeled_data)

    if not load_indices:
        labeled_indices = np.random.choice(indices, config.size_labeled_data, replace=False)
        unlabeled_indices = indices
        np.savetxt(intermidiate_path, labeled_indices)
    else:
        labeled_indices = np.loadtxt(intermidiate_path).astype(int)
        unlabeled_indices = indices[~mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_loader = DataLoader(config, training_set, np.arange(len(training_set))[~mask], config.train_batch_size)

    return labeled_loader, unlabeled_loader, dev_loader, special_loader, training_set


def get_cifar_loaders(config, load_indices=False):
    
    root_path = "/home/zhanghuayi01/lancet/LANCET/"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10(root_path + '/data/cifar', train=True, download=True, transform=transform)
    dev_set = CIFAR10(root_path + '/data/cifar', train=False, download=True, transform=transform)

    np.random.seed(1)
    indices = np.arange(len(training_set))
    
    intermidiate_path = "{}cifar_labeled_indices_{}.txt".format(config.save_dir, config.size_labeled_data)

    if load_indices:
        labeled_indices = np.loadtxt(intermidiate_path).astype(int)
        mask = np.zeros(50000, dtype=bool)
        mask[labeled_indices] = True
        unlabeled_indices = np.arange(50000)[~mask]

        labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)

    else:
        mask = np.zeros(indices.shape[0], dtype=np.bool)
        labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
        mask[np.random.choice(np.arange(indices.shape[0]), size=config.size_labeled_data, replace=False)] = True
        labeled_indices, unlabeled_indices = indices[mask], indices[~mask]
        # save the indices
        np.savetxt(intermidiate_path, labeled_indices)

    print('labeled size', labeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_loader = DataLoader(config, training_set, np.arange(len(training_set))[~mask], config.train_batch_size)

    return labeled_loader, unlabeled_loader, dev_loader, special_loader, training_set


def get_cifar100_loaders(config, load_indices=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    training_set = CIFAR100('../cifar', train=True, download=True, transform=transform_train)
    dev_set = CIFAR100('../cifar', train=False, download=True, transform=transform)

    np.random.seed(1)
    indices = np.arange(len(training_set))

    if load_indices:
        labeled_indices = np.loadtxt("labeled_indices.txt").astype(int)
        mask = np.zeros(50000, dtype=bool)
        mask[labeled_indices] = True
        unlabeled_indices = np.arange(50000)[~mask]

        labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)

    else:
        mask = np.zeros(indices.shape[0], dtype=np.bool)
        labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
        mask[np.random.choice(np.arange(indices.shape[0]), size=config.size_labeled_data, replace=False)] = True
        labeled_indices, unlabeled_indices = indices[mask], indices[~mask]
        # save the indices
        np.savetxt("labeled_indices.txt", labeled_indices)

    print('labeled size', labeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_loader = DataLoader(config, training_set, np.arange(len(training_set))[~mask], config.train_batch_size)

    return labeled_loader, unlabeled_loader, dev_loader, special_loader, training_set


def get_speechcommand_loaders(config, load_indices=False):
    # loading data
    target_class = np.arange(10)
    train_dataset = GCommandLoader(config.train_path, window_size=config.window_size, window_stride=config.window_stride,
                                   window_type=config.window_type, normalize=config.normalize, target_class=target_class)
    # unl_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=60,
    #                                          pin_memory=config.cuda, sampler=None)
    # unl_loader_ = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=60,
    #                                           pin_memory=config.cuda, sampler=None)

    unl_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1,
                                             pin_memory=config.cuda, sampler=None)
    unl_loader_ = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1,
                                              pin_memory=config.cuda, sampler=None)

    label_cnt = [train_dataset[i][1] for i in range(len(train_dataset))]
    print(np.unique(label_cnt, return_counts=True))

    num_labels = config.size_labeled_data
    np.random.seed(1)
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    labeled_indices = indices[:num_labels]
    unlabeled_indices = indices[num_labels:]
    label_cnt = [train_dataset[i][1] for i in labeled_indices]
    print("Speechcommand Dataset: The number of objects in each class:", np.unique(label_cnt, return_counts=True))

    intermidiate_path = "/home/zhanghuayi01/lancet/LANCET/speechcommand_intermidiate_result/{}/".format(config.method)
    np.savetxt(intermidiate_path + "speechcommand_labeled_indices_{}.txt".format(len(labeled_indices)), labeled_indices)
    print ("Saving {} labeled indices of speechcommand dataset at {}.".format(len(labeled_indices), intermidiate_path))


    lab_loader = DataLoader(config, train_dataset, indices=labeled_indices, batch_size=config.batch_size)

    valid_dataset = GCommandLoader(config.valid_path, window_size=config.window_size, window_stride=config.window_stride,
                                   window_type=config.window_type, normalize=config.normalize, target_class=target_class)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=None,
        num_workers=20, pin_memory=config.cuda, sampler=None)

    test_dataset = GCommandLoader(config.test_path, window_size=config.window_size, window_stride=config.window_stride,
                                  window_type=config.window_type, normalize=config.normalize, target_class=target_class)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.test_batch_size, shuffle=None,
        num_workers=20, pin_memory=config.cuda, sampler=None)

    return lab_loader, unl_loader, valid_loader, test_loader, train_dataset

def get_har_loaders(config, load_indices=False):
    
    train_dataset, test_dataset = create_har_dataset()
    
    unlabeled_loader = DataLoader(config, train_dataset, indices=np.arange(len(train_dataset)), batch_size=100)
    labeled_loader = DataLoader(config, train_dataset,
                                indices=np.random.choice(np.arange(len(train_dataset)), size=config.size_labeled_data, replace=False),
                                batch_size=100)
    dev_loader = DataLoader(config, test_dataset, indices=np.arange(len(test_dataset)), batch_size=100)

    return labeled_loader, unlabeled_loader, dev_loader

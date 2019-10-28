# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Function for using CIFAR10 dataset
#

import os
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def load_cifar(num_classes=10, batch_size=128):

    num_train = 50000
    indices = list(range(num_train))
    num_val = 5000

    torch.manual_seed(301)
    torch.cuda.manual_seed(301)
    np.random.seed(301)
    random.seed(301)

    val_idx = np.random.choice(indices, size=num_val, replace=False)
    print(val_idx)
    train_idx = list(set(indices) - set(val_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    num_test = 10000
    test_idx = list(range(num_test))
    test_sampler = SubsetRandomSampler(test_idx)

    if num_classes == 10:
        dataset = torchvision.datasets.CIFAR10
    else:
        dataset = torchvision.datasets.CIFAR100

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    data_dir = os.path.expanduser('~/hyperdatasets/cifar')

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              sampler=train_sampler, num_workers=2, pin_memory=True)

    val_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=False, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                              sampler=val_sampler, num_workers=2, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              sampler=test_sampler, num_workers=2, pin_memory=True)

    classes = range(num_classes)

    return train_loader, val_loader, test_loader, classes


if __name__ == '__main__':
    print(load_cifar())


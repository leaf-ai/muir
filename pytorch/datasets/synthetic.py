# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Function for loading synthetic dataset
#

import pickle
import numpy as np
import os
import torch
import torch.utils.data

def load_synthetic(batch_size=15, num_val=5, task_index=0, noisy=False):

    data_dir = os.path.expanduser('~/hyperdatasets') + '/synthetic'

    if noisy:
        noisy_str = 'noisy_'
    else:
        noisy_str = ''

    train_X_file = data_dir + '/' + 'train_X.pkl'
    test_X_file = data_dir + '/' + 'test_X.pkl'
    train_Y_file = data_dir + '/' + noisy_str + 'train_Y.pkl'
    test_Y_file = data_dir + '/' + noisy_str + 'test_Y.pkl'

    with open(train_X_file, 'rb') as f:
        train_X = pickle.load(f, encoding='latin1')[task_index]
    with open(test_X_file, 'rb') as f:
        test_X = pickle.load(f, encoding='latin1')[task_index]
    with open(train_Y_file, 'rb') as f:
        train_Y = pickle.load(f, encoding='latin1')[task_index]
    with open(test_Y_file, 'rb') as f:
        test_Y = pickle.load(f, encoding='latin1')[task_index]

    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    np.random.seed(301)
    indices = range(train_X.shape[0])
    val_idx = np.random.choice(indices, size=num_val, replace=False)
    train_idx = list(set(indices) - set(val_idx))

    val_X = train_X[val_idx]
    train_X = train_X[train_idx]
    val_Y = train_Y[val_idx]
    train_Y = train_Y[train_idx]

    print(train_X.shape)
    print(train_Y.shape)
    print(val_X.shape)
    print(val_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    train_X = torch.Tensor(train_X)
    train_Y = torch.Tensor(train_Y).view(-1, 1)
    val_X = torch.Tensor(val_X)
    val_Y = torch.Tensor(val_Y).view(-1, 1)
    test_X = torch.Tensor(test_X)
    test_Y = torch.Tensor(test_Y).view(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_Y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    classes = None

    return train_loader, val_loader, test_loader, None

if __name__ == '__main__':
    print(load_synthetic())

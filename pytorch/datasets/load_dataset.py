# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Utility for loading datasets
#

import sys

from datasets.cifar import load_cifar
from datasets.crispr import load_crispr_genomic
from datasets.synthetic import load_synthetic
from datasets.wikitext2 import load_wikitext2

def load_dataset(task_config):

    dataset = task_config['dataset']
    batch_size = task_config['batch_size']

    print("Loading", dataset)

    if dataset == 'cifar10':
        trainloader, valloader, testloader, classes = load_cifar(10, batch_size)

    elif dataset == 'cifar100':
        trainloader, valloader, testloader, classes = load_cifar(100, batch_size)

    elif dataset == 'crispr_genomic':
        trainloader, valloader, testloader, classes = load_crispr_genomic(batch_size)

    elif dataset == 'synthetic':
        num_val = task_config.get('num_val', 5)
        task_index = task_config.get('task_index', 0)
        noisy = task_config.get('noisy', False)
        trainloader, valloader, testloader, classes = load_synthetic(batch_size=batch_size,
                                                                     num_val=num_val,
                                                                     task_index=task_index,
                                                                     noisy=noisy)

    elif dataset == 'wikitext2':
        trainloader, valloader, testloader, classes = load_wikitext2(batch_size=batch_size)

    else:
        print("Please select a valid dataset")
        sys.exit(0)

    return {'name': dataset,
            'train_loader': trainloader,
            'val_loader': valloader,
            'test_loader': testloader}


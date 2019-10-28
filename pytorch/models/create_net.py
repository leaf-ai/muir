# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Utility function for creating networks.
#

import sys

from models.deepbind import DeepBind
from models.lenet import LeNet
from models.linear_model import LinearModel
from models.lstm_language_model import LSTMLanguageModel
from models.wide_resnet import WideResnet

def create_net(task_config, projector_config):

    model = task_config['model']
    output_size = task_config['output_size']
    context_size = projector_config.get('context_size', None)
    block_in = projector_config.get('block_in', None)
    block_out = projector_config.get('block_out', None)

    print("Creating", model)

    if context_size > 0:
        hyper = True
        hyperlayers = ['conv2']
    else:
        hyper = False
        hyperlayers = []

    if model == 'deepbind':
        num_filters = task_config.get('num_filters', 16)
        hidden_dim = task_config.get('hidden_dim', 32)
        net = DeepBind(context_size, block_in, block_out, {'context_size': 100}, hyper,
                        filters=num_filters, hidden_units=hidden_dim)

    elif model == 'linear_model':
        input_size = task_config.get('input_dim', 20)
        net = LinearModel(context_size, block_in, block_out,
                          input_dim=input_size, output_dim=output_size, hyper=hyper)

    elif model == 'lstm_language_model':
        layer_size = task_config.get('layer_size', 32)
        net = LSTMLanguageModel(context_size, block_in, block_out,
                                ninp=layer_size, nhid=layer_size, hyper=hyper)

    elif model == 'wide_resnet':
        N = task_config.get('N', 6)
        k = task_config.get('k', 1)
        num_classes = output_size
        net = WideResnet(context_size, block_in, block_out, N, k, num_classes, hyper)

    elif model == 'lenet':
        if context_size > 0:
            hyperlayers = ['conv2', 'fc1', 'fc2']
        net = LeNet(context_size, block_in, block_out, hyperlayers)

    else:
        print("Please select a valid model kind")
        sys.exit(0)

    return net


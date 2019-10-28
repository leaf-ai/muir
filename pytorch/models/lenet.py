# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Hypermodule implementation of LeNet in pytorch
#

import torch.nn as nn
from torch.nn import functional as F
from layers.hyperconv2d import HyperConv2d
from layers.hyperlinear import HyperLinear
from muir.hyper_utils import set_layer_weights

class LeNet(nn.Module):
    def __init__(self, context_size, block_in, block_out, hyperlayers=[]):
        super(LeNet, self).__init__()

        self.hyperlayers = []

        if 'conv1' in hyperlayers:
            self.conv1 = HyperConv2d(3, 16, 5, context_size, block_in, block_out, padding=2)
            self.hyperlayers.append(self.conv1)
        else:
            self.conv1 = nn.Conv2d(3, 16, 5, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        if 'conv2' in hyperlayers:
            self.conv2 = HyperConv2d(16, 16, 5, context_size, block_in, block_out, padding=2)
            self.hyperlayers.append(self.conv2)
        else:
            self.conv2 = nn.Conv2d(16, 16, 5, padding=1)

        if 'fc1' in hyperlayers:
            self.fc1 = HyperLinear(16 * 6 * 6, 128, context_size, block_in, block_out)
            self.hyperlayers.append(self.fc1)
        else:
            self.fc1 = nn.Linear(16 * 6 * 6, 128)

        if 'fc2' in hyperlayers:
            self.fc2 = HyperLinear(128, 96, context_size, block_in, block_out)
            self.hyperlayers.append(self.fc2)
        else:
            self.fc2 = nn.Linear(128, 96)

        if 'fc3' in hyperlayers:
            self.fc3 = HyperLinear(96, 10, context_size, block_in, block_out)
            self.hyperlayers.append(self.fc3)
        else:
            self.fc3 = nn.Linear(96, 10)

        self.num_projectors = sum([l.num_projectors for l in self.hyperlayers])
        print(self.hyperlayers)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_weights(self, params):
        assert params.size(0) == self.num_projectors
        set_layer_weights(params, self.hyperlayers)

if __name__ == '__main__':
    print(LeNet(1, 2, 4))
    print(LeNet(1, 2, 4, hyperlayers=['fc1', 'fc2']))
    print(LeNet(1, 2, 4, hyperlayers=['conv2', 'fc1', 'fc2']))
    net = LeNet(1, 2, 4, hyperlayers=[])

    from model_utils import count_parameters
    print(count_parameters(net))

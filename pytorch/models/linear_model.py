# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Hypermodule implementation of simple linear model
#

import torch.nn as nn
from layers.hyperlinear import HyperLinear
from muir.hyper_utils import set_layer_weights

class LinearModel(nn.Module):
    def __init__(self, context_size=1, block_in=20, block_out=1,
                 input_dim=20, output_dim=1, hyper=False, bias=False):
        super(LinearModel, self).__init__()

        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out
        self.hyper = hyper

        self.hyperlayers = []

        if hyper:
            self.fc = HyperLinear(input_dim, output_dim,
                                  context_size, block_in, block_out,
                                  bias=bias)
            self.hyperlayers.append(self.fc)
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=bias)

        self.num_projectors = sum([l.num_projectors for l in self.hyperlayers])

    def set_weights(self, params):
        assert params.size(0) == self.num_projectors
        set_layer_weights(params, self.hyperlayers)

    def forward(self, x):
        return self.fc(x)

if __name__ == '__main__':
    net = LinearModel()
    print(net)


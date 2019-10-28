# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/optimize/LICENSE.
#
# Class for pytorch projectors, i.e., hypermodules.
#

import math
import torch.nn as nn

class Projector(nn.Module):

    def __init__(self, context_size, block_in, block_out, bias=True):
        super(Projector, self).__init__()
        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out
        self.bias = bias
        self.projector = nn.Linear(context_size, block_in*block_out, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.context_size)
        nn.init.normal_(self.projector.weight, 0, std)
        if self.bias:
            nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        x = self.projector(x).view(self.block_out, self.block_in)
        return x

if __name__ == '__main__':
    print(Projector(10, 20, 30))


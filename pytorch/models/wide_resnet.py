# Copyright (c) 2019 Cognizant Digital Business.
#
# Code derived from https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py,
# issued under the following license:
#
# BSD 2-Clause License
#
# Copyright (c) 2016, Sergey Zagoruyko
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#
# Wide resnet with hypermodule reparameterization options.
#

import torch.nn as nn
from torch.nn import functional as F
from layers.hyperconv2d import HyperConv2d
from layers.hyperlinear import HyperLinear
from muir.hyper_utils import set_layer_weights

class WideResnet(nn.Module):
    def __init__(self, context_size, block_in, block_out, N, k, num_classes,
                 hyper=True, spatial_sharing=False):
        super(WideResnet, self).__init__()

        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out

        self.N = N
        self.k = k

        self.hyper = hyper
        self.hyperlayers = []

        self.conv0 = nn.Conv2d(3, 16, 3, bias=False)
        nn.init.kaiming_normal_(self.conv0.weight, nonlinearity='relu')

        self.group0 = self.create_group(16, 16 * k, stride=1)
        self.group1 = self.create_group(16 * k, 32 * k, stride=2)
        self.group2 = self.create_group(32 * k, 64 * k, stride=2)

        self.final_bn = nn.BatchNorm2d(64 * k)
        self.fc = nn.Linear(64 * k, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

        self.num_projectors = sum([l.num_projectors for l in self.hyperlayers])

    def create_group(self, in_channels, out_channels, stride=1):
        """
        Creates a group of blocks of the given kind.
        Appends any created hyperlayers to self.hyperlayers to track num_projectors.
        """

        group = nn.ModuleDict()

        if in_channels != out_channels:
            if self.hyper:
                resampler = HyperConv2d(in_channels, out_channels, 3,
                                        self.context_size, self.block_in, self.block_out,
                                        stride=stride)
                self.hyperlayers.append(resampler)
            else:
                resampler = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
            group['resampler'] = resampler

        block_list = nn.ModuleList()
        block0 = self.create_block(in_channels, out_channels, stride=stride)
        block_list.append(block0)
        for i in range(1, self.N):
            block = self.create_block(out_channels, out_channels, stride=1)
            block_list.append(block)
        group['block_list'] = block_list

        return group

    def create_block(self, in_channels, out_channels, stride=1):

        bn0 = nn.BatchNorm2d(in_channels)
        if self.hyper:
            conv0 = HyperConv2d(in_channels, out_channels, 3,
                                self.context_size, self.block_in, self.block_out,
                                stride=stride)
        else:
            conv0 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        bn1 = nn.BatchNorm2d(out_channels)
        if self.hyper:
            conv1 = HyperConv2d(out_channels, out_channels, 3,
                                self.context_size, self.block_in, self.block_out,
                                stride=1)
        else:
            conv1 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        block = nn.ModuleDict({'bn0': bn0,
                               'conv0': conv0,
                               'bn1': bn1,
                               'conv1': conv1})

        if self.hyper:
            self.hyperlayers.append(conv0)
            self.hyperlayers.append(conv1)

        return block

    def set_weights(self, params):
        assert params.size(0) == self.num_projectors
        set_layer_weights(params, self.hyperlayers)

    def apply_group(self, group, x):
        if 'resampler' in group:
            y = group['resampler'](x)
        else:
            y = x

        for block in group['block_list']:
            z = block['bn0'](x)
            z = F.relu(z)
            z = block['conv0'](z)
            z = block['bn1'](z)
            z = F.relu(z)
            z = block['conv1'](z)

            x = z + y
            y = x

        return x

    def forward(self, x):

        x = self.conv0(x)

        x = self.apply_group(self.group0, x)
        x = self.apply_group(self.group1, x)
        x = self.apply_group(self.group2, x)

        x = self.final_bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 8, 1, 0)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    net = WideResnet(4, 16, 16, 6, 1, 10)
    print(net)
    print(net.num_projectors)

    net = WideResnet(4, 16, 16, 6, 1, 10, hyper=False)
    from model_utils import count_parameters
    print(count_parameters(net))

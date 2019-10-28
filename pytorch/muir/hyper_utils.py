# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Utility functions for implementing and using hyperlayers
#

import torch
from muir.projector import Projector

def create_projectors(num_projectors, context_size, block_in, block_out, norm=None):
    projectors = []
    for p in range(num_projectors):
        projector = Projector(context_size, block_in, block_out, norm=None)
        projectors.append(projector)
    return projectors

def pack_kernel(contexts, projectors, in_features, out_features, block_in, block_out):

    # Create blocks
    blocks = []
    for context, projector in zip(contexts, projectors):
        block = projector(context)
        blocks.append(block)

    # Concatenate blocks
    num_block_rows = out_features / block_out
    num_block_cols = in_features / block_in
    block_rows = []
    for r in range(num_block_rows):
        start = r * num_block_cols
        end = start + num_block_cols
        block_row = torch.cat(blocks[start:end], dim=1)
        block_rows.append(block_row)
    kernel = torch.cat(block_rows, dim=0)

    return kernel

def pack_dense(params, input_size, output_size):

    num_in_blocks = int(input_size / params.size(2))
    chunks = params.chunk(num_in_blocks, dim=0)
    weight = torch.cat(chunks, dim=2).view(output_size, input_size)
    return weight

def pack_conv1d(params, input_size, output_size, kernel_size):

    matrix = pack_dense(params, input_size, output_size * kernel_size)
    chunks = matrix.chunk(kernel_size, dim=0)
    weight = torch.stack(chunks, dim=2)
    return weight

def pack_conv2d(params, input_size, output_size, kernel_size):

    conv1d = pack_conv1d(params, input_size, output_size, kernel_size * kernel_size)
    weight = conv1d.view(output_size, input_size, kernel_size, kernel_size)
    return weight

def set_layer_weights(params, hyperlayers):
    param_idx = 0
    for layer in hyperlayers:
        layer.set_weight(params[param_idx:param_idx+layer.num_projectors])
        param_idx += layer.num_projectors


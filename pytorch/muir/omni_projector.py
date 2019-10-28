# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Class for a single-pass parameter generator.
#

import math
import torch
import torch.nn as nn

class OmniProjector(nn.Module):

    def __init__(self, context_size, block_in, block_out, num_blocks, num_candidates, context,
                 bias=True, alpha=None, ignore_context=False, frozen_context=False):
        super(OmniProjector, self).__init__()

        self.context_size = context_size
        self.block_in = block_in
        self.block_out = block_out
        self.num_blocks = num_blocks
        self.num_candidates = num_candidates
        self.bias = bias
        self.ignore_context = ignore_context

        # alpha is the initial probability evenly distributed to new candidates upon assembly.
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = alpha
            self.soft_init_constant = math.log(alpha) - math.log(num_candidates-1) - math.log(1.-alpha)

        self.projectors = nn.Parameter(torch.Tensor(num_blocks, block_in * block_out, context_size))

        if self.bias:
            self.biases = nn.Parameter(torch.Tensor(num_blocks, block_in * block_out, 1))

        self.soft_weights = nn.Parameter(torch.Tensor(num_candidates, num_blocks))

        self.context = nn.Parameter(torch.Tensor(context))
        if frozen_context:
            self.context.requires_grad = False

        self.map_tensors = None

        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.context_size)
        nn.init.normal_(self.projectors, 0, std)
        if self.bias:
            nn.init.zeros_(self.biases)
        nn.init.zeros_(self.soft_weights)

    def reactivate_projectors(self, projector_idxs, device):
        if len(projector_idxs) == 0:
            return
        std = 1. / math.sqrt(self.context_size)
        with torch.no_grad():
            init_values = torch.randn(len(projector_idxs),
                                      self.block_in * self.block_out,
                                      self.context_size) * std
            init_values = init_values.to(device)
            self.projectors[projector_idxs, :, :] = init_values

    def assemble_projectors(self, projector_map):
        self.num_candidates = max([len(projector_list) for projector_list in projector_map])
        self.map_tensors = []
        for i in range(self.num_candidates):
            mapped_projectors = []
            for projector_list in projector_map:
                if len(projector_list) <= i:
                    projector_idx = projector_list[0]
                else:
                    projector_idx = projector_list[i]
                mapped_projectors.append(projector_idx)
            map_tensor = torch.LongTensor(mapped_projectors)
            self.map_tensors.append(map_tensor)

        if self.alpha is None:
            nn.init.zeros_(self.soft_weights)
        else:
            nn.init.zeros_(self.soft_weights[0])
            nn.init.constant_(self.soft_weights[1:], self.soft_init_constant)


    def get_candidate_params(self, candidate_idx, context):
        if self.ignore_context:
            return torch.squeeze(
                        self.projectors[self.map_tensors[candidate_idx], :, :]
                    ).view(self.num_blocks, self.block_out, self.block_in)

        elif self.bias:
            return torch.squeeze(
                        torch.bmm(self.projectors[self.map_tensors[candidate_idx], :, :], context)
                        + self.biases[self.map_tensors[candidate_idx], :, :]
                    ).view(self.num_blocks, self.block_out, self.block_in)

        else:
            return torch.squeeze(
                        torch.bmm(self.projectors[self.map_tensors[candidate_idx], :, :], context)
                    ).view(self.num_blocks, self.block_out, self.block_in)

    def forward(self):
        if self.num_candidates == 1:
            return self.get_candidate_params(0, self.context)

        candidate_probs = torch.softmax(self.soft_weights, dim=0).view(
            self.num_candidates, self.num_blocks, 1, 1)
        params = self.get_candidate_params(0, self.context) * candidate_probs[0]
        for i in range(1, self.num_candidates):
            params += self.get_candidate_params(i, self.context) * candidate_probs[i]
        return params


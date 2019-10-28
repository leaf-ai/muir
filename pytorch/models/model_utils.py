# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/LICENSE.
#
# Util functions for model processing.
#

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

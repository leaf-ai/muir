# muir

This repository provides an implementation of the algorithm introduced in [Modular Universal Reparameterization: Deep Multi-task Learning across Diverse Domains, NeurIPS 2019](https://arxiv.org/pdf/1906.00097.pdf).

## Installation
```
cd muir
mkdir results
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:~/muir/pytorch/
```

## Datasets

The code assumes datasets are downloaded and placed in `~/hyperdatasets/<dataset_name>`, e.g., `~/hyperdatasets/cifar` and `~/hyperdatasets/wikitext2`.

Dataset files for the synthetic dataset are included directly in `muir/datasets/synthetic`.

Dataset files for Cifar can be downloaded directly with PyTorch.

Dataset files for WikiText-2 can be downloaded from https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/.

Dataset files for CRISPR binding prediction can be downloaded from: http://nn.cs.utexas.edu/pages/research/crispr-binding-prediction.tar.gz.

## Running Optimization

```
cd muir/pytorch/muir
python optimize.py --experiment_name <exp_name> --config <config_file> --device <device_id>
```

`experiment_name` is the name of the experiment and can be anything. Experiment launch time information will be appended to this name.

`config` is a path to the config file. For example configs, see `muir/pytorch/configs`.

`device` is the name of the device for running torch, e.g., `cpu`, `cuda:0`, `cuda:1`, ...

Results for the experiment will be saved to a directory with the experiments name in `muir/results`.

## Implementing new Experiments

To use a new architecture, a model class can be implemented that replaces layers with hyperlayers (see `muir/pytorch/models/` for examples).

Currently, layers supported for reparameterization by hypermodules are fully-connected, conv2d, conv1d, and LSTM (see `muir/pytorch/layers/`). These can be extended to more layer types by following the examples there.

To use a new dataset, it can be implemented to follow the interface of the examples in `muir/pytorch/datasets/`.


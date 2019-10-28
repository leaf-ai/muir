# Copyright (c) 2019 Cognizant Digital Business.
# Issued under this Academic Public License: github.com/leaf-ai/muir/pytorch/muir/LICENSE.
#
# Entrypoint for optimization in muir
#

import argparse
import pickle
import os
import time
import sys
sys.path.insert(0, os.getcwd())
import yaml

from copy import deepcopy
from past.builtins import basestring
from shutil import copyfile

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.load_dataset import load_dataset
from models.create_net import create_net
from muir.load_utils import load_config
from muir.omni_projector import OmniProjector


def get_initial_state(num_projectors, num_locations, mode='separate'):
    if mode == 'separate':
        assert num_projectors == num_locations
        return [[loc] for loc in range(num_locations)]
    elif mode == 'perfect':
        state = []
        for i in range(30):
            p = i / 10
            state.append([p])
        print(state)
        return state
    elif mode == 'random':
        state = np.random.choice(range(num_locations), size=num_locations, replace=True)
        state = [int(state[i]) for i in range(num_locations)]
        print(state)
        return state


def get_new_state(curr_state, num_projectors, num_loc, num_cand,
                  reactivation_probability):
    """
    Add a num_cand challengers in each of num_loc locations.
    """
    new_state = deepcopy(curr_state)
    num_locations = len(curr_state)
    if num_loc < 1:
        num_loc = int(np.ceil(num_locations * num_loc))
    else:
        num_loc = int(num_loc)
    active_projectors = get_active_projectors(curr_state)
    print("Active Projectors:", len(active_projectors))
    inactive_projectors = set(range(num_projectors)) - active_projectors
    target_locations = np.random.choice(range(num_locations), size=num_loc, replace=False)
    reactivated_projectors = set()
    for target_location in target_locations:
        for cand in range(num_cand):
            select_active = (np.random.random() > reactivation_probability) \
                            or (num_locations == len(active_projectors))
            if select_active:
                source_location = np.random.randint(num_locations)
                source_projector = curr_state[source_location][0]
            else:
                source_projector = np.random.choice(list(inactive_projectors))
                reactivated_projectors.add(source_projector)

            new_state[target_location].append(source_projector)
    return new_state, list(reactivated_projectors)


def get_active_projectors(curr_state):
    active_projectors = set()
    for loc in curr_state:
        for projector in loc:
            active_projectors.add(projector)
    return active_projectors


def refine_state(state, omni_projector, select='best'):
    """
    Reduce state to a single projector for each location.
    """

    candidate_probs = torch.softmax(omni_projector.soft_weights, dim=0).squeeze()
    if len(candidate_probs.size()) == 2:
        candidate_probs = candidate_probs.t()
    else:
        candidate_probs = candidate_probs.unsqueeze(dim=0).t()

    refined_state = []
    for i, loc in enumerate(state):
        if len(loc) == 1:
            best_projector = loc[0]

        elif select == 'best':

            cumulative_probs = {}
            for j, proj in enumerate(loc):
                prob = candidate_probs[i][j].item()
                if proj not in cumulative_probs:
                    cumulative_probs[proj] = 0
                cumulative_probs[proj] += prob

            best_projector = -1
            best_score = -1
            for proj  in cumulative_probs:
                score = cumulative_probs[proj]
                if score > best_score:
                    best_score = score
                    best_projector = proj
            best_idx = loc.index(best_projector)

        elif select == 'random':
            best_idx = np.random.randint(len(loc))
            best_projector = loc[best_idx]
        refined_state.append([best_projector])

    return refined_state


def get_experiment_dir(experiment_name):

    results_root = os.path.expanduser('~/muir/results/')
    experiment_dir = results_root + '/' + experiment_name
    return experiment_dir


def setup_experiment_dir(experiment_name, config_file):

    experiment_dir = get_experiment_dir(experiment_name)

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    else:
        print("Please choose a new experiment name")
        sys.exit(0)

    copyfile(config_file, '{}/{}'.format(experiment_dir, config_file.split('/')[-1]))

    scores_file = experiment_dir + '/scores.txt'
    with open(scores_file, 'w') as f:
        f.write("[Train Losses] [Val Losses] [Val Errs] [Test Losses] [Test Errs]\n")
    states_file = experiment_dir + '/states.txt'
    with open(states_file, 'w') as f:
        pass


def write_result(experiment_name, scores, curr_state):

    experiment_dir = get_experiment_dir(experiment_name)

    score_names = ['train_losses', 'val_losses', 'val_errs', 'test_losses', 'test_errs']
    line = ' '.join([str(scores[score]) for score in score_names]) + '\n'
    scores_file = '{}/scores.txt'.format(experiment_dir)
    with open(scores_file, 'a') as f:
        f.write(line)
    line = '{}\n'.format(curr_state)
    states_file = '{}/states.txt'.format(experiment_dir)
    with open(states_file, 'a') as f:
        f.write(line)


def set_params(params, nets):
    start = 0
    for net in nets:
        end = start + net.num_projectors
        net.set_weights(params[start:end])
        start = end


def create_optimizer(nets, omni_projector, config):

    param_list = []
    for i, module in enumerate(nets):
        param_list.append({'params': module.parameters(), 'name': 'net_{}'.format(i)})

    if omni_projector:
        param_list.append({'params': omni_projector.projectors, 'name': 'projectors'})

        if hasattr(omni_projector, 'biases'):
            param_list.append({'params': omni_projector.biases, 'name': 'biases'})
        param_list.append({'params': omni_projector.soft_weights,
                          'lr': config['optimization'].get('soft_lr', 0.1),
                          'name': 'soft_weights'})
        param_list.append({'params': omni_projector.context, 'name': 'context'})

    optimizer_name = config['training'].get('optimizer', 'adam')
    lr = config['training'].get('lr', 0.001)
    if optimizer_name == 'adam':
        optimizer = optim.Adam(param_list, lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(param_list, lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(param_list, lr=lr)
    else:
        print("Please choose a valid optimizer")
        sys.exit(0)

    return optimizer


def create_criteria(task_configs):
    criteria = []
    for task_config in task_configs:
        loss = task_config['loss']
        if loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif loss == 'mse':
            criterion = nn.MSELoss()
        else:
            print("Please choose a valid loss")
            sys.exit(0)
        criteria.append(criterion)
    return criteria


def run_nets(datasets, nets, criteria, optimizer, omni_projector, config, device):

    train_loaders = [dataset['train_loader'] for dataset in datasets]
    train_iters = [dataset['train_iter'] for dataset in datasets]
    val_loaders = [dataset['val_loader'] for dataset in datasets]
    test_loaders = [dataset['test_loader'] for dataset in datasets]

    print("Training")
    train_losses = train(train_loaders, train_iters, nets, criteria, optimizer, omni_projector,
                         config['optimization'].get('steps_per_generation'), device)
    print("Evaluating")
    val_losses, val_errs = test(val_loaders, nets, criteria, device)
    print("Testing")
    test_losses, test_errs = test(test_loaders, nets, criteria, device)

    return {'train_losses': train_losses,
            'val_losses': val_losses,
            'val_errs': val_errs,
            'test_losses': test_losses,
            'test_errs': test_errs}

def aggregate_context(nets):
    print("Aggregating Context")
    context_list = []
    for net in nets:
        for layer in net.hyperlayers:
            context_list.append(layer.context)
    context = torch.cat(context_list, dim=0)
    aggregate_context = torch.tensor(context, requires_grad=True)
    return aggregate_context


def train(train_loaders, train_iters, nets, criteria, optimizer, omni_projector, steps, device):

    [net.train() for net in nets]
    if omni_projector:
        omni_projector.train()
    running_losses = [0.0 for net in nets]

    train_start = time.time()
    for i in range(steps):
        iter_start = time.time()
        optimizer.zero_grad()

        if omni_projector:
            set_params(omni_projector(), nets)

        for j, (train_loader, net, criterion) in enumerate(zip(train_loaders, nets, criteria)):
            start_time = time.time()
            try:
                batch = next(train_iters[j])
            except:
                train_iters[j] = iter(train_loader)
                batch = next(train_iters[j])
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels = batch.text, batch.label.long()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            retain_variables = (j != len(nets) - 1)
            loss.backward(retain_graph=retain_variables)

            elapsed_time = time.time() - start_time
            running_losses[j] += loss.item()
            print('dataset: %5d, step: %5d, secs: %.2f, loss: %.3f, %.3f' %
                  (j, i + 1, elapsed_time, running_losses[j] / (i+1), loss.item()))

            if (i + 1) % len(train_loader) == 0:
                train_iters[j] = iter(train_loader)

        optimizer.step()

    print(running_losses[j] / (i+1))

    train_losses = [running_loss / steps for running_loss in running_losses]
    return train_losses


def test(test_loaders, nets, criteria, device):

    test_losses = []
    test_errs = []
    for i, (test_loader, net, criterion) in enumerate(zip(test_loaders, nets, criteria)):
        if hasattr(net, 'init_hidden'):
            print("Init Hidden")
            net.init_hidden()
        print("Testing", i)
        net.eval()
        test_loss = 0
        total = 0
        correct = 0
        classify = (isinstance(criterion, nn.CrossEntropyLoss))
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs, labels = batch.text, batch.label.long()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                if classify:
                    pred = outputs.max(1, keepdim=True)[1]
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                total += labels.size(0)

        test_loss /= total
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, total,
            100. * correct / float(total)))
        test_err = 1. - (float(correct) / total)

        test_losses.append(test_loss)
        test_errs.append(test_err)

    print("Mean Loss:", np.mean(test_losses))

    return test_losses, test_errs


def reset_soft_weight_optimizer_state(optimizer):
    """
    Reset optimizer state of soft weights so they are not misapplied.
    """
    for group in optimizer.param_groups:
        if 'name' in group and group['name'] == 'soft_weights':
            for p in group['params']:
                optimizer.state[p] = {}


def optimize(experiment_name, datasets, nets, omni_projector, config, device):

    [net.to(device) for net in nets]
    if omni_projector:
        omni_projector.to(device)

    optimizer = create_optimizer(nets, omni_projector, config)

    criteria = create_criteria(config['tasks'])
    if omni_projector:
        num_projectors = omni_projector.num_blocks
    else:
        num_projectors = 0

    init_mode = config['optimization'].get('init_mode', 'separate')
    curr_state = get_initial_state(num_projectors, num_projectors, init_mode)

    best_score = float("inf")
    best_state = deepcopy(curr_state)

    # Decide how many initial meta-iterations to train with the default parameterization
    # before beginning optimization.
    burn_in = config['optimization'].get('burn_in', 0)

    for gen in range(config['optimization']['num_generations']):
        print("Generation",gen)
        print(experiment_name)

        # Generate
        if (num_projectors > 0) and (gen >= burn_in):
            print("GENERATING")
            curr_state, reactivated_projectors = get_new_state(curr_state, num_projectors,
                                                       config['optimization']['generate_percentage'],
                                                       config['optimization']['alternatives_per_generation'],
                                                       config['optimization']['reactivation_probability'])
            omni_projector.reactivate_projectors(reactivated_projectors, device)

        if omni_projector:
            omni_projector.assemble_projectors(curr_state)
            reset_soft_weight_optimizer_state(optimizer)

        # Evaluate
        scores = run_nets(datasets, nets, criteria, optimizer, omni_projector, config, device)

        # Refine
        if omni_projector and (config['optimization']['alternatives_per_generation'] > 0):
            curr_state = refine_state(curr_state, omni_projector,
                                      config['optimization']['selection_method'])

        # Check how well we're doing
        target_metric = config['optimization'].get('target_metric', 'val_losses')
        if 'target_index' in config['optimization']:
            target_index = config['optimization']['target_index']
            curr_score = scores[target_metric][target_index]
        else:
            curr_score = np.mean(scores[target_metric])

        save_best = config['optimization'].get('save_best', False)
        if curr_score < best_score:

            print("New best score:", curr_score)
            best_score = curr_score
            best_state = deepcopy(curr_state)

            if save_best:
                print("Saving best")
                save_best_state(experiment_name, best_state, omni_projector, nets, optimizer)

        # Record
        write_result(experiment_name, scores, curr_state)


    # Revert state and perform any final training.
    final_training_generations = config['optimization'].get('final_training_generations', 0)
    if final_training_generations > 0:
        print("Loading best")
        omni_projector, nets, curr_state, optimizer = load_best_state(
                        experiment_name, omni_projector, nets, optimizer)

        if omni_projector:
            omni_projector.assemble_projectors(curr_state)
    for gen in range(final_training_generations):
        print("Final Generation",gen)

        # Evaluate
        scores = run_nets(datasets, nets, criteria, optimizer, omni_projector, config, device)

        # Record
        write_result(experiment_name, scores, curr_state)


def save_best_state(experiment_name, best_state, omni_projector, nets, optimizer):
    experiment_dir = get_experiment_dir(experiment_name)

    omni_projector_path = '{}/omni_projector.pth'.format(experiment_dir)
    torch.save(omni_projector.state_dict(), omni_projector_path)

    for i, net in enumerate(nets):
        net_path = '{}/net_{}.pth'.format(experiment_dir, i)
        torch.save(net.state_dict(), net_path)

    optimizer_path = '{}/optimizer.pth'.format(experiment_dir)
    torch.save(optimizer.state_dict(), optimizer_path)

    state_path = '{}/state.pkl'.format(experiment_dir)
    with open(state_path, 'wb') as f:
        pickle.dump(best_state, f)


def load_best_state(experiment_name, omni_projector, nets, optimizer):
    experiment_dir = get_experiment_dir(experiment_name)

    omni_projector_path = '{}/omni_projector.pth'.format(experiment_dir)
    omni_projector.load_state_dict(torch.load(omni_projector_path))

    for i, net in enumerate(nets):
        net_path = '{}/net_{}.pth'.format(experiment_dir, i)
        net.load_state_dict(torch.load(net_path))

    optimizer_path = '{}/optimizer.pth'.format(experiment_dir)
    optimizer.load_state_dict(torch.load(optimizer_path))

    state_path = '{}/state.pkl'.format(experiment_dir)
    with open(state_path, 'rb') as f:
        best_state = pickle.load(f)

    return omni_projector, nets, best_state, optimizer


def setup_and_run(experiment_name, config, device):

    datasets = [load_dataset(task_config) for task_config in config['tasks']]
    for dataset in datasets:
        dataset['train_iter'] = iter(dataset['train_loader'])
    nets = [create_net(task_config, config['projectors']) for task_config in config['tasks']]
    num_projectors = sum([net.num_projectors for net in nets])
    if num_projectors != 0:
        context = aggregate_context(nets)
        bias = config['projectors'].get('bias', False)
        alpha = config['optimization'].get('alpha', None)
        ignore_context = config['projectors'].get('ignore_context', False)
        frozen_context = config['projectors'].get('frozen_context', False)
        omni_projector = OmniProjector(config['projectors']['context_size'],
                                       config['projectors']['block_in'],
                                       config['projectors']['block_out'],
                                       num_projectors,
                                       config['optimization']['alternatives_per_generation'] + 1,
                                       context,
                                       bias=bias,
                                       alpha=alpha,
                                       ignore_context=ignore_context,
                                       frozen_context=frozen_context)
    else:
        omni_projector = None
    optimize(experiment_name, datasets, nets, omni_projector, config, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimize alignment of hypermodules")

    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()
    setup_experiment_dir(args.experiment_name, args.config)
    config = load_config(args.config)
    setup_and_run(args.experiment_name, config, args.device)

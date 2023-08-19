#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs / utils.utils
@File    ：utils.py
@Author  ：LiangL. Yan
"""
import os
import torch


def print_network(model):
    """Print out the information of network."""
    nums_params = 0
    for p in model.parameters():
        nums_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(nums_params))


def print_net_params(model):
    """Print out the information of network params."""
    for p in model.parameters():
        print(p)


def save_model(model_save_dir, save_name, model):
    """Save the model."""
    path = os.path.join(model_save_dir, save_name)
    torch.save(model.state_dict(), path)
    print('Saved model checkpoints into {}...'.format(model_save_dir))


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


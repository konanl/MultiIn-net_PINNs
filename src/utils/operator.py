'''
Descripttion: 
version: 
Author: Yoking.L
Date: 2023-08-19 14:31:51
LastEditTime: 2023-08-19 14:36:35
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs / utils.operator 
@File    ：operator.py
@Author  ：LiangL. Yan
"""
import torch
import numpy as np


def gradients(x, y, order=1):
    """Computer the gradient : Dy / Dx."""
    if order == 1:
        return torch.autograd.grad(y, x,
                                   grad_outputs=torch.ones_like(y),
                                   create_graph=True,
                                   # retain_graph=True, only_inputs=True
                                   )[0]
    else:
        return gradients(gradients(x, y), x, order=order - 1)


# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


def l2_relative_error(y_true, y_pred):
    """L2 norm relative error."""
    if isinstance(y_pred, np.ndarray):
        return np.linalg.norm((y_true - y_pred)) / np.linalg.norm(y_true)
    else:
        y_pred = np.array(y_pred.detach().numpy())
        return np.linalg.norm((y_true - y_pred)) / np.linalg.norm(y_true)

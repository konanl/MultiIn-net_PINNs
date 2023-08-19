#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs / models.Base
@File    ：Base.py
@Author  ：LiangL. Yan
"""
import torch
import torch.nn as nn


## init operate ##
## see: https://github.com/nodtem66/PINN_Implicit_SDF/blob/328cdabf8654e839e6d2032a3b72c60ec6efcb0d/models/Base.py
def activation_name(activation: nn.Module) -> str:
    if activation is nn.Tanh:
        return 'tanh'
    elif activation is nn.ReLU or activation is nn.ELU or activation is nn.GELU:
        return 'relu'
    elif activation is nn.SELU:
        return 'selu'
    elif activation is nn.LeakyReLU:
        return 'leaky_relu'
    elif activation is nn.Sigmoid:
        return 'sigmoid'
    return 'linear'


def linear_layer_with_init(width, height, init=nn.init.xavier_uniform_, activation=None) -> nn.Linear:
    linear = nn.Linear(width, height)
    if init is None or activation is None:
        return linear
    init(linear.weight, gain=nn.init.calculate_gain(activation_name(activation)))
    return linear


class PINN(nn.Module):
    """
        The Basic model of Physics Informed Neural Networks(PINNs).
            :parameter
                _basic_net : the base neural network in PINNs.

            :arg
                In fact, you need to reload the loss.
    """
    def __init__(self, input_dim=2, output_dim=3, Netdeep=7, active=nn.Tanh()):
        super(PINN, self).__init__()

        self.in2out = [input_dim, output_dim]
        self.deep = Netdeep

        self.active = active

        self._basic_net = nn.Sequential()

        # Build Neural Network
        self._build_network()

    def _build_network(self):
        # self._basic_net = nn.Sequential()
        self._basic_net.add_module('Linear_layer_1', nn.Linear(self.in2out[0], 30))
        self._basic_net.add_module('Tanh_layer_1', self.active)
        for num in range(2, self.deep):
            self._basic_net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))
            self._basic_net.add_module('Tanh_layer_%d' % (num), self.active)
        self._basic_net.add_module('Linear_layer_final', nn.Linear(30, self.in2out[1]))
        # return self._basic_net

    def _basic_equation(self, **kwargs):
        """The PDE or others control equations of the questions you want to solve."""
        pass

    def loss_residual(self, **kwargs):
        """The loss of PDE/ODE."""
        pass

    def loss_ic(self, **kwargs):
        """The loss of Initialization condition."""
        pass

    def loss_bc(self, **kwargs):
        """The loss of Boundary condition."""
        pass

    def forward(self, x):
        return self._basic_net(x)


class PINNFree(nn.Module):
    """
        This Basic model of Physics Informed Neural Networks(PINNs) is more easier to expand.
    """
    def __init__(self, net):
        super(PINNFree, self).__init__()

        self._basic_net = net

    def forward(self, x):
        return self._basic_net(x)

    def _basic_equation(self, **kwargs):
        """The PDE or others control equations of the questions you want to solve."""
        pass

    def loss_residual(self, **kwargs):
        """The loss of PDE/ODE."""
        pass

    def loss_ic(self, **kwargs):
        """The loss of Initialization condition."""
        pass

    def loss_bc(self, **kwargs):
        """The loss of Boundary condition."""
        pass

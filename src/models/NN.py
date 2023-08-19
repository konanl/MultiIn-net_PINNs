#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs / models.NN
@File    ：NN.py
@Author  ：LiangL. Yan
"""
import torch
import torch.nn as nn


class FCNet(nn.Module):
    """FC Neural Network."""
    def __init__(self, layers, w_init=True, active=nn.Tanh()):
        super(FCNet, self).__init__()

        # Parameters
        self.depth = len(layers) - 1
        self.active = active

        # Layers list
        layer_list = list()
        for layer in range(self.depth - 1):
            layer_list.append(
                nn.Linear(layers[layer], layers[layer+1])
            )
            layer_list.append(active)
        layer_list.append(nn.Linear(layers[-2], layers[-1]))

        # Net
        self.main = nn.Sequential(*layer_list)

        # Initialize parameters
        if w_init:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.main(x)


class Network(nn.Module):
    def __init__(self, layers, active=nn.Tanh()):
        super(Network, self).__init__()

        self.layers = layers
        self.active = active

        # Input Layer
        self.input_layer = None

        # Hidden Layer
        self.hidden_layer = None

        # Output Layer
        self.output_layer = nn.Sequential()

        # convert
        self.convert_layer = nn.Sequential(nn.Linear(layers[1], layers[-1]))

        self._build_network()

    def _build_network(self):
        # Input
        self.input_layer = nn.Sequential(nn.Linear(self.layers[0], self.layers[1]))
        # Hidden
        layer_list = list()
        for layer in range(2, len(self.layers) - 2):
            layer_list.append(nn.Linear(self.layers[layer], self.layers[layer+1]))
            layer_list.append(self.active)
        self.hidden_layer = nn.Sequential(*layer_list)
        # Output
        self.output_layer.add_module('Linear_layer_final', nn.Linear(self.layers[-2], self.layers[-1]))

    def forward(self, x):
        input = self.input_layer(x)
        input_pred = self.convert_layer(input)
        hidden = self.hidden_layer(input)
        hidden_pred = self.convert_layer(hidden)
        output = self.output_layer(hidden)

        return input_pred, hidden_pred, output


class ResNet(nn.Module):
    """Deep Residual Learning for Image Recognition[arXiv:1512.03385]."""
    def __init__(self, backbone_layers, residual_layers):
        super(ResNet, self).__init__()

        # Check input and output
        # if backbone_layers[1] != residual_layers[0]:
        #     raise AssertionError("Input size of backbone net and residual net do not match!")
        if len(residual_layers) > 0 and backbone_layers[-2] != residual_layers[-1]:
            raise AssertionError("Output size of backbone net and residual net do not match!")

        # layers
        self.res_layers = residual_layers
        self.backbone_layers = backbone_layers

        # Input layer
        self.input_layer = FCNet(self.backbone_layers[:2])

        # Backbone network.
        self.backbone = FCNet(self.backbone_layers[1:-1])

        # Residual network.
        self.residual = self._build_resnet()

        # Output layer
        self.output = FCNet(self.backbone_layers[-2:])

    def _build_resnet(self):
        """Build Res Connection."""
        if len(self.res_layers) == 0:
            return None
        layer_list = list()
        layer_list.append(nn.Linear(self.backbone_layers[1], self.res_layers[0]))
        layer_list.append(nn.Tanh())

        if len(self.res_layers) <= 1:
            return nn.Sequential(*layer_list)

        for layer in range(1, len(self.res_layers)-1):
            layer_list.append(nn.Linear(self.res_layers[layer], self.res_layers[layer+1]))
            layer_list.append(nn.Tanh())

        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.input_layer(x)
        b = self.backbone(x)

        if self.residual != None:
            r = self.residual(x)
            x = r + b
        x = x + b

        return self.output(x)


class UNet(nn.Module):

    def __init__(self, unet_link_layers, center_layers):
        super(UNet, self).__init__()

        self.unet_link_layers = unet_link_layers
        self.center_layers = center_layers

        # Network
        # [2,10,20,[20, 20],20,10,1]
        self.layer1 = nn.Sequential(
            nn.Linear(self.unet_link_layers[0], self.unet_link_layers[1]),
            nn.Tanh()
        )

        self.layer2 = nn.Sequential(nn.Linear(self.unet_link_layers[1], self.unet_link_layers[2]), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(self.unet_link_layers[2], self.center_layers[0]), nn.Tanh())

        self.center = nn.Sequential(nn.Linear(self.unet_link_layers[2], self.center_layers[0]), nn.Tanh(),
                                    nn.Linear(self.center_layers[0], self.center_layers[1]), nn.Tanh(),
                                    nn.Linear(self.center_layers[1], self.unet_link_layers[3]), nn.Tanh()
                                    )

        self.layer4 = nn.Sequential(nn.Linear(self.unet_link_layers[3], self.unet_link_layers[4]), nn.Tanh())
        # self.layer5 = nn.Sequential(nn.Linear(self.unet_link_layers[5], self.unet_link_layers[6]), nn.Tanh())

        self.output = nn.Sequential(nn.Linear(self.unet_link_layers[4], self.unet_link_layers[-1]))

    def forward(self, x):

        in1 = self.layer1(x) # [N, 10]
        in2 = self.layer2(in1) # [N, 20]
        # in3 = self.layer3(in2) # [N, 20]

        center = self.center(in2) # [N, 20]

        out1 = self.layer4(center + in2) # [N, 20]
        # out2 = self.layer5(out1 + in2) # [N, 10]
        out = self.output(out1 + in1)

        return out


class MultiInNet(nn.Module):
    def __init__(self, trunk_net_layers, branch_net_layers, active=nn.Tanh(), w_init=True):
        super(MultiInNet, self).__init__()

        assert trunk_net_layers[0] == branch_net_layers[-1], "The Branch Net output should equal to Trunk Net input!!!"
        assert trunk_net_layers[0] == branch_net_layers[0], "The Branch Net`s input != Trunk Net`s!!!"

        # Parameters
        self.depth_trunk = len(trunk_net_layers) - 1
        self.depth_branch = len(branch_net_layers) - 1
        self.active = active

        # Layers list
        # Trunk
        layer_list_trunk = list()
        for layer in range(self.depth_trunk - 2):
            layer_list_trunk.append(
                nn.Linear(trunk_net_layers[layer], trunk_net_layers[layer+1])
            )
            layer_list_trunk.append(active)
        # layer_list_trunk.append(nn.Linear(trunk_net_layers[-3], trunk_net_layers[-2]))

        # Branch
        layer_list_branch = list()
        for layer in range(self.depth_branch - 1):
            layer_list_branch.append(
                nn.Linear(branch_net_layers[layer], branch_net_layers[layer+1])
            )
            layer_list_branch.append(active)
        layer_list_branch.append(nn.Linear(branch_net_layers[-2], branch_net_layers[-1]))

        # output layer
        self.output_layer = nn.Sequential(nn.Linear(trunk_net_layers[-2], trunk_net_layers[-1]))

        # connect
        self.connect = nn.Sequential(nn.Linear(branch_net_layers[-1], trunk_net_layers[-2]))

        # Net
        self.trunk_main = nn.Sequential(*layer_list_trunk)
        self.branch_main = nn.Sequential(*layer_list_branch)

        # self.connect_main = nn.Sequential(nn.Linear())

        # Initialize parameters
        if w_init:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, x):

        y_branch = self.branch_main(x)
        # print(y_branch.shape)
        # x = self.active(y_branch + x)
        x = torch.mul(y_branch, x)
        y_trunk = self.trunk_main(x)

        y_branch = self.connect(y_branch)
        # print(y_branch.shape, y_trunk.shape)
        y = self.output_layer(y_branch * y_trunk)
        return y


class MultiInNetCorrection(nn.Module):
    def __init__(self, input_dim, output_dim, neuron_num, trunk_net, branch_net):
        super(MultiInNetCorrection, self).__init__()

        self._sigmod = nn.Sigmoid()
        self._tanh = nn.Tanh()
        self._ReLU = nn.ReLU()

        ########## Trunk ##########
        self.trunk_input = nn.Linear(input_dim, neuron_num)
        self.trunk_hidden = FCNet(trunk_net)
        # self.trunk_link = nn.Linear(neuron_num, neuron_num)

        ########## Branch ##########
        self.branch_input = nn.Linear(input_dim, neuron_num)
        self.branch_hidden = FCNet(branch_net)
        self.branch_output = nn.Linear(neuron_num, neuron_num)

        ########## __OUTPUT__ ##########
        self.out2input = nn.Linear(neuron_num, input_dim)
        self.__OUTPUT__ = nn.Linear(neuron_num, output_dim)

    def forward(self, x):

        # __STEP 1.
        x_branch = self.branch_input(x)
        x_0 = self._sigmod(self.out2input(x_branch))
        x = x * x_0
        # __STEP 2.
        x_branch = self.branch_hidden(x_branch)
        x = self._tanh(self.trunk_input(x) + self._sigmod(x_branch))
        # __STEP 3.
        x_branch = self.branch_output(x_branch)
        x = self._tanh(self.trunk_hidden(x) + self._ReLU(x_branch))
        # __STEP 4.
        OUT = self.__OUTPUT__(self._ReLU(x * x_branch))

        return OUT


if __name__ == '__main__':

    # model = InverseResNet([1, 20, 20, 20, 1], [20, 20, 30, 20])
    # x = torch.Tensor(100, 1)
    # print(x.shape)
    # y = model(x)
    # print(y.shape)

    # net = UNet([2,10,20,20,10,1], [20,20])
    # net = ResNet([2,20,20,20,20,1], [20,20])
    # net = MultiInNet([2, 20, 20, 20, 20, 1], [2, 20, 20, 2])
    net = MultiInNetCorrection(2, 1, 20, [20, 20, 20], [20, 20, 20])
    # a = torch.Tensor(100, 2)
    # y = net(a)
    # print(y.shape, net)
    print(net)
    a = torch.rand(1000, 2)
    y = net(a)
    print(y.shape)

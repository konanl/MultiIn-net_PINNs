#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs Poisson
@File    ：poisson1d.py
@Author  ：LiangL. Yan
"""

"""
    The Poisson is:
        -d2u / dx2 = f(X), x-[-L, L]
    with boundary condition:
        u(-L) = u(L) = 0
    we set L = 2*sqrt(pi), and the exact u(x) = sin(x**2)
    then, the d2u / dx2 = 2cos(x**2) - 4x**2*sin(x**2), x-[-L, L],
    
"""

import sys
sys.path.append("../..")
import numpy as np
from SALib.sample import sobol_sequence
from src.models.Base import PINNFree
from src.models.NN import FCNet, UNet, ResNet, MultiInNet, MultiInNetCorrection
import torch
import torch.nn as nn
from src.utils.operator import gradients, to_numpy, l2_relative_error
import time
from src.utils.utils import save_model, model_info
import scipy.io
import os
import matplotlib.pyplot as plt
from src.utils.visualizer import set_defult_rcParams
import re


# Seeds
torch.manual_seed(123456)
np.random.seed(123456)

# Params
L = 2 * np.sqrt(np.pi)


def u(x):
    """Exact 1D Poisson Function Solution."""
    return np.sin(x**2)


def du(x):
    """Exact Gradient of 1D Poisson Function Solution."""
    if isinstance(x, torch.Tensor):
        return 2*torch.cos(x**2) - 4*(x**2)*torch.sin(x**2)
    return 2*np.cos(x**2) - 4*(x**2)*np.sin(x**2)


def gen_traindata(N, method='uniform'):

    if method == 'uniform':
        Nx = 2 * N
        x = np.linspace(-L, L, Nx, endpoint=False)
    elif method == 'sobol':
        a = sobol_sequence.sample(2*N, 1)
        x = a[:, 0:1] * 2 * L - L
    else:
        x = np.random.rand(2 * N) * 2 * L - L

    X = np.reshape(x, (-1, 1))
    return X.astype(np.float32)


def gen_testdata(N, method='uniform'):

    if method == 'uniform':
        Nx = 2 * N
        x = np.linspace(-L, L, Nx, endpoint=False)
    elif method == 'sobol':
        a = sobol_sequence.sample(2*N, 1)
        x = a[:, 0:1] * 2 * L - L
    else:
        x = np.random.rand(2 * N) * 2 * L - L

    y = u(x)
    X = np.reshape(x, (-1, 1))
    Y = np.reshape(y, (-1, 1))
    return X.astype(np.float32), Y.astype(np.float32)


def gen_bcdata(N):
    """u(-L) = u(L) = 0"""
    Nx = 1 * N
    xl = np.linspace(-L, -L, Nx, endpoint=False)
    xr = np.linspace(L, L, Nx, endpoint=False)
    x = np.hstack((xl, xr))
    np.random.shuffle(x)

    X = np.reshape(x, (-1, 1))
    return X.astype(np.float32)


class Poisson1D(PINNFree):

    def __init__(self, net):
        super(Poisson1D, self).__init__(net)

    def loss_residual(self, x):
        """u + d2u / dx2 = 0"""
        u = self.forward(x)

        dudx = gradients(x, u)
        d2udx2 = gradients(x, dudx)

        f = d2udx2 - du(x)

        return (f**2).mean()

    def loss_bc(self, x):

        u = self.forward(x)

        return (u**2).mean()


def train_batch(epoch, train_x, train_x_bc, model, optimizer):
    """Train a model in one epoch."""
    LOSS = []

    def closure():
        optimizer.zero_grad()

        # Compute loss.
        loss_r = model.loss_residual(train_x)
        loss_bc = model.loss_bc(train_x_bc)

        w_r, w_bc = 1, 1
        loss = w_r * loss_r + w_bc * loss_bc

        LOSS.append(loss_r)
        LOSS.append(loss_bc)
        LOSS.append(loss)

        print(f'epoch {epoch} loss_pde:{loss_r:.2e},  loss_bc:{loss_bc:.2e}')
        loss.backward()
        return loss

    Loss = optimizer.step(closure)

    loss_value = Loss.item() if not isinstance(Loss, float) else Loss
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'epoch {epoch} loss:{loss_value:.4e}, lr:{lr:.4e}')
    print('--------------------------------------------------')

    return LOSS[0], LOSS[1], LOSS[2]


def main():

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Train data
    ## residual
    num_train_points = 1000  # double
    train_x = gen_traindata(num_train_points, method='sobol')
    train_x = torch.tensor(train_x, dtype=torch.float32, requires_grad=True)
    train_x = train_x.to(device)

    ## boundary condition data
    train_x_bc = gen_bcdata(num_train_points // 2)
    train_x_bc = torch.tensor(train_x_bc, dtype=torch.float32, requires_grad=True)
    train_x_bc = train_x_bc.to(device)

    # Initialize neural network.
    # net = FCNet([1, 20, 20, 20, 1])
    # net = ResNet([1, 20, 20, 20, 20, 1], [20, 20])
    # net = UNet([1, 10, 20, 20, 10, 1], [20, 20])
    # net = MultiInNet([1, 20, 20, 20, 1], [1, 20, 1])
    net = MultiInNetCorrection(1, 1, 20, [20, 20], [20, 20])
    model = Poisson1D(net).to(device)
    # print(model)

    # training parameters.
    epochs, lr = 8000, 1e-3

    # optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.90)

    # loss log.
    LOSS = []

    # start time
    print("## Start Training...")
    tic = time.time()

    # epochs_step1 = 1250
    # for epoch in range(1, epochs_step1+1):
    #     # train
    #     model.train()
    #     for k, v in model.named_parameters():
    #         re_network = re.compile(r'^_basic_net.output_layer')
    #         if re_network.match(k): # or k == '_basic_net.branch_main.6.weight' or k == '_basic_net.branch_main.6.bias':
    #             v.requires_grad = True
    #         else:
    #             v.requires_grad = False
    #
    #     epoch_train_loss_r, epoch_train_loss_bc, epoch_train_loss \
    #         = train_batch(epoch, train_x, train_x_bc, model, optimizer)
    #
    #     LOSS.append([epoch,
    #                  to_numpy(epoch_train_loss_r),
    #                  to_numpy(epoch_train_loss_bc),
    #                  to_numpy(epoch_train_loss)])

        # scheduler.step()

    # epochs_step2 = 1500
    # for epoch in range(epochs_step1+1, epochs_step2+1):
    #     # train
    #     model.train()
    #     for k, v in model.named_parameters():
    #         re_network = re.compile(r'^_basic_net.output_layer')
    #         if re_network.match(k):
    #             v.requires_grad = False
    #         else:
    #             v.requires_grad = True
    #
    #     epoch_train_loss_r, epoch_train_loss_bc, epoch_train_loss \
    #         = train_batch(epoch, train_x, train_x_bc, model, optimizer)
    #
    #     LOSS.append([epoch,
    #                  to_numpy(epoch_train_loss_r),
    #                  to_numpy(epoch_train_loss_bc),
    #                  to_numpy(epoch_train_loss)])

        # scheduler.step()

    epochs_step2 = 0
    for epoch in range(epochs_step2+1, epochs+1):
        # train
        model.train()

        for k, v in model.named_parameters():
            v.requires_grad = True

        epoch_train_loss_r, epoch_train_loss_bc, epoch_train_loss \
            = train_batch(epoch, train_x, train_x_bc, model, optimizer)

        LOSS.append([epoch,
                     to_numpy(epoch_train_loss_r),
                     to_numpy(epoch_train_loss_bc),
                     to_numpy(epoch_train_loss)])

        # scheduler.step()

    toc = time.time()
    print(f'Total training time: {toc - tic}')
    print('\nTrain Done......')

    # Save the network parameters
    save_name = 'pinn-mc-wo' + str(epochs) + '.pth'
    model_save_dir = './model/'
    save_model(model_save_dir, save_name, model._basic_net)

    # Evaluate on the whole computational domain
    # Test data
    vaild_x, vaild_u = gen_testdata(100)
    vaild_x = torch.tensor(vaild_x, dtype=torch.float32)
    vaild_u = torch.tensor(vaild_u, dtype=torch.float32)
    vaild_x, vaild_u = vaild_x.to(device), vaild_u.to(device)

    u_pred = model(vaild_x)

    # mean l2_error
    l2_error = l2_relative_error(to_numpy(vaild_u), to_numpy(u_pred))

    # Save training loss.
    scipy.io.savemat(f'./work/loss/Poisson1d-pinn-mc-loss-wo.mat', {'loss': LOSS})

    # Save predict result
    scipy.io.savemat(f'./work/Poisson1d-pinn-mc-pred-wo.mat', {'x': to_numpy(vaild_x),
                                                  'u_true': to_numpy(vaild_u),
                                                  'u_pred': to_numpy(u_pred),
                                                  # 'error_u': error_u,
                                                  # 'error_pde': error_pde,
                                                  'l2_re': l2_error
                                                      }
                     )


if __name__ == '__main__':

    # Create directories if not exist.
    log_dir = './work/loss/'
    model_save_dir = "./model/"

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # main()

    #################################################
    ###################### Plot #####################
    #################################################
    set_defult_rcParams()

    # Plot
    ## Loss
    LOSS = scipy.io.loadmat("./work/loss/Poisson1d-pinn-m-loss.mat")['loss']
    epochs, loss_r, loss_bc, loss = LOSS[:, 0], LOSS[:, 1], LOSS[:, 2], LOSS[:, 3]
    plt.plot(epochs, loss_r, 'bs', markersize=1, label='$loss_r$')
    plt.plot(epochs, loss_bc, '+', color='purple', markersize=1, label='$loss_{bc}$')
    plt.plot(epochs, loss, 'r^', markersize=1, label='$loss$')
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    plt.ylim(0, 0.4)
    plt.xlim(1000, 2000)
    plt.axhline(15e-2, c='black', linestyle='--')
    plt.axvline(1000, c='black', linestyle='--')
    # plt.axvline(2000, c='black', linestyle='--')
    # plt.axvline(4000, c='black', linestyle='--')
    # plt.axvline(8000, c='black', linestyle='--')
    plt.xlabel('Iterations', fontsize=20)
    # plt.title("Loss of Multi Step Train UNet PINNs", fontsize=12)
    # plt.title("Loss of MultiInNet PINNs", fontsize=18)
    plt.title("Loss of MultiInNet PINNs", fontsize=13)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='best')
    # plt.savefig('./work/fig/poisson1d-pinn-m-loss-wo-ab.png', bbox_inches='tight', dpi=600)
    plt.show()

    ## l2
    data0 = scipy.io.loadmat('./work/Poisson1d-pinn-pred.mat')
    data1 = scipy.io.loadmat('./work/Poisson1d-pinn-n-pred.mat')
    data2 = scipy.io.loadmat('./work/Poisson1d-pinn-r-pred-wo.mat')
    data3 = scipy.io.loadmat('./work/Poisson1d-pinn-u-pred-wo.mat')
    data4 = scipy.io.loadmat('./work/Poisson1d-pinn-m-pred.mat')

    x0, u_true0, u_pred0, l2_error0 = data0['x'], data0['u_true'], data0['u_pred'], data0['l2_re']
    x1, u_true1, u_pred1, l2_error1 = data1['x'], data1['u_true'], data1['u_pred'], data1['l2_re']
    x2, u_true2, u_pred2, l2_error2 = data2['x'], data2['u_true'], data2['u_pred'], data2['l2_re']
    x3, u_true3, u_pred3, l2_error3 = data3['x'], data3['u_true'], data3['u_pred'], data3['l2_re']
    x4, u_true4, u_pred4, l2_error4 = data4['x'], data4['u_true'], data4['u_pred'], data4['l2_re']

    print(f"Vanilla MSE:{((u_true0 - u_pred0)**2).mean()}, L2-error:{l2_error0}")
    print(f"Multi Step Train MSE:{((u_true1 - u_pred1) ** 2).mean()}, L2-error:{l2_error1}")
    print(f"Multi Step Train Res MSE:{((u_true2 - u_pred2) ** 2).mean()}, L2-error:{l2_error2}")
    print(f"Multi Step Train Unet MSE:{((u_true3 - u_pred3) ** 2).mean()}, L2-error:{l2_error3}")
    print(f"Multi Step Train Multi MSE:{((u_true4 - u_pred4) ** 2).mean()}, L2-error:{l2_error4}")

    ## predict
    plt.plot(x0, u_true0, color='black', marker='o', markersize=1, label='$Exact$')
    plt.plot(x0, u_pred0, 'r^', markersize=2, label='$Vanilla\:PINNs$')
    plt.plot(x1, u_pred1, 'bs', markersize=2, label='$Multi\:Step\:Train\:PINNs$')
    plt.plot(x3, u_pred3, 'g+', markersize=2, label='$UNet\:PINNs$')
    plt.plot(x2, u_pred2, 'y-', markersize=2, label='$ResNet\:PINNs$')
    plt.plot(x4, u_pred4, 'o-', markersize=2, label='$MultiInNet\:PINNs$')
    plt.xlabel('x', fontsize=20)
    plt.ylabel('u', fontsize=20)
    plt.title("Predict", fontsize=20)
    # plt.rcParams.update({'font.size': 12})
    plt.legend(loc='best')
    # plt.savefig('./work/fig/poisson1d-pred-vs-u.png', bbox_inches='tight', dpi=600)
    plt.show()
    
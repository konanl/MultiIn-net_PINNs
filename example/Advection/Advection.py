#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs Advection
@File    ：Advection.py
@Author  ：LiangL. Yan
"""

"""
    The Advection is:
        pu / dt + pu / px = 0, x \in [0, 1], t \in [0, 0.5]
    with initial condition is :
        u(0, x) = 2 * sin(pi * x)
    with boundary condition is :
        u(t, 0) = -2 * sin(pi * t), u(t, 1) = 2 * sin(pi * t)
    And the Exact solution is :
        u(t, x) = 2 * sin(pi * (x - t))   

"""

import sys
sys.path.append("../..")
import numpy as np
from SALib.sample import sobol_sequence
from src.models.Base import PINNFree
from src.models.NN import FCNet, UNet, ResNet, MultiInNet, MultiInNetCorrection
import torch
from src.utils.operator import gradients, to_numpy, l2_relative_error
import time
from src.utils.utils import save_model, model_info
import scipy.io
import os
import matplotlib.pyplot as plt
from src.utils.visualizer import set_defult_rcParams
import re


def u(x, t):
    return 2 * np.sin(np.pi * (x - t))


def gen_traindata(N, method='uniform'):

    if method == 'uniform':
        Nt = int((N / 2) ** 0.5)
        Nx = 2 * Nt
        x = np.linspace(0, 1, Nx, endpoint=False)
        t = np.linspace(0, 0.5, Nt, endpoint=False)
        xx, tt = np.meshgrid(x, t)
    elif method == 'sobol':
        a = sobol_sequence.sample(N, 2)
        xx = a[:, 0:1]
        tt = a[:, 1:2] * 0.5
    else:
        xx = np.random.random(N)
        tt = np.random.random(N) * 0.5

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    return X.astype(np.float32)


def gen_testdata(N):

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 0.5, N)
    xx, tt = np.meshgrid(x, t)

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u(X[..., 0], X[..., 1])

    return X.astype(np.float32), y.astype(np.float32)


def gen_initdata(N, method='uniform'):
    """u(0, x) = 2 * sin(pi * x)."""
    if method == 'uniform':
        Nt = int((N / 2) ** 0.5)
        Nx = 2 * Nt
        x = np.linspace(0, 1, Nx, endpoint=False)
        t = np.linspace(0, 0, Nt, endpoint=False)
        xx, tt = np.meshgrid(x, t)
    elif method == 'sobol':
        a = sobol_sequence.sample(N, 2)
        xx = a[:, 0:1]
        tt = a[:, 1:2] * 0
    else:
        xx = np.random.random(N)
        tt = np.random.random(N) * 0.0

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    y = []
    for item in X:
        y.append(2 * np.sin(np.pi * item[0]))
    y = np.reshape(y, (X.shape[0], -1))

    return X.astype(np.float32), y.astype(np.float32)


def gen_bcdata(N):
    """ u(t, 0) = -2 * sin(pi * t), u(t, 1) = 2 * sin(pi * t)."""
    Nt = int((N / 2) ** 0.5)
    Nx = 2 * Nt
    xl = np.linspace(0, 0, Nx, endpoint=False)
    xr = np.linspace(1, 1, Nx, endpoint=False)
    x = np.hstack((xl, xr))
    np.random.shuffle(x)

    t = np.linspace(0, 0.5, Nt, endpoint=False)

    xx, tt = np.meshgrid(x, t)

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    y = []
    for item in X:
        y.append((int(item[0]) * (2 * np.sin(np.pi * item[1]))) + int(1 - item[0]) * -2 * np.sin(np.pi * item[1]))
    y = np.reshape(y, (X.shape[0], -1))

    return X.astype(np.float32), y.astype(np.float32)


class Advection(PINNFree):
    
    def __init__(self, net):
        super(Advection, self).__init__(net)

    def loss_residual(self, x):
        """pu / dt + pu / px = 0."""
        u = self.forward(x)

        du = gradients(x, u)
        dudx = du[:, 0]
        dudt = du[:, 1]

        f = dudx + dudt

        return (f**2).mean()

    def loss_ic(self, x):
        """u(0, x) = 2 * sin(pi * x)."""
        u = self.forward(x[0])

        # f = u - 2 * torch.sin(torch.pi * x[:, 0])

        return ((u - x[1])**2).mean()

    def loss_bc(self, x):
        """u(t, 0) = -2 * sin(pi * t), u(t, 1) = 2 * sin(pi * t)."""
        u = self.forward(x[0])

        return ((u - x[1])**2).mean()


def train_batch(epoch, train_x, train_ic, train_bc, model, optimizer):
    """Train a model in one epoch."""
    LOSS = []

    def closure():
        optimizer.zero_grad()

        # Compute loss.
        loss_r = model.loss_residual(train_x)
        loss_bc = model.loss_bc(train_bc)
        loss_ic = model.loss_ic(train_ic)

        w_r, w_ic, w_bc = 1, 1, 1
        loss = w_r * loss_r + w_bc * loss_bc + w_ic * loss_ic

        LOSS.append(loss_r)
        LOSS.append(loss_ic)
        LOSS.append(loss_bc)
        LOSS.append(loss)

        print(f'epoch {epoch} loss_pde:{loss_r:.2e},  loss_ic:{loss_ic:.2e}, loss_bc:{loss_bc:.2e}')
        loss.backward()
        return loss

    Loss = optimizer.step(closure)

    loss_value = Loss.item() if not isinstance(Loss, float) else Loss
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print(f'epoch {epoch} loss:{loss_value:.4e}, lr:{lr:.4e}')
    print('--------------------------------------------------')

    return LOSS[0], LOSS[1], LOSS[2], LOSS[3]


def main():

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Train data
    ## residual
    num_train_points = 1000  # double
    train_x = gen_traindata(num_train_points, method='sobol')
    train_x = torch.tensor(train_x, dtype=torch.float32, requires_grad=True)
    train_x = train_x.to(device)

    ## initial condition
    train_x_ic, train_u_ic = gen_initdata(num_train_points // 2)
    train_x_ic = torch.tensor(train_x_ic, dtype=torch.float32, requires_grad=True)
    train_u_ic = torch.tensor(train_u_ic, dtype=torch.float32, requires_grad=True)
    train_u_ic = train_u_ic.to(device)
    train_x_ic = train_x_ic.to(device)
    train_ic = [train_x_ic, train_u_ic]

    ## boundary condition data
    train_x_bc, train_u_bc = gen_bcdata(num_train_points // 2)
    train_x_bc = torch.tensor(train_x_bc, dtype=torch.float32, requires_grad=True)
    train_u_bc = torch.tensor(train_u_bc, dtype=torch.float32, requires_grad=True)
    train_x_bc = train_x_bc.to(device)
    train_u_bc = train_u_bc.to(device)
    train_bc = [train_x_bc, train_u_bc]

    # Initialize neural network.
    ## Vanilla PINN
    # net = FCNet([2, 20, 20, 20, 1])
    ## Inverse Res PINN
    # net = InverseResNet([2, 20, 20, 20, 1], [20, 20])
    # u-Net
    # net = UNet([2, 20, 10, 10, 20, 1], [10, 10])
    # net = ResNet([2, 20, 20, 20, 1], [20, 20])
    # net = MultiInNet([2, 20, 20, 20, 1], [2, 20, 20, 2])
    net = MultiInNetCorrection(2, 1, 20, [20, 20, 20, 20], [20, 20, 20])
    model = Advection(net).to(device)

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
    #         re_network = re.compile(r'^_basic_net.output')
    #         if re_network.match(k):
    #             v.requires_grad = True
    #         else:
    #             v.requires_grad = False
    #
    #     epoch_train_loss_r, epoch_train_loss_ic, epoch_train_loss_bc, epoch_train_loss \
    #         = train_batch(epoch, train_x, train_ic, train_bc, model, optimizer)
    #
    #     LOSS.append([epoch,
    #                  to_numpy(epoch_train_loss_r),
    #                  to_numpy(epoch_train_loss_ic),
    #                  to_numpy(epoch_train_loss_bc),
    #                  to_numpy(epoch_train_loss)])
    #
    #     # scheduler.step()
    #
    # epochs_step2 = 2500
    # for epoch in range(epochs_step1 + 1, epochs_step2 + 1):
    #     # train
    #     model.train()
    #     for k, v in model.named_parameters():
    #         re_network = re.compile(r'^_basic_net.output')
    #         if re_network.match(k):
    #             v.requires_grad = False
    #         else:
    #             v.requires_grad = True
    #
    #     epoch_train_loss_r, epoch_train_loss_ic, epoch_train_loss_bc, epoch_train_loss \
    #         = train_batch(epoch, train_x, train_ic, train_bc, model, optimizer)
    #
    #     LOSS.append([epoch,
    #                  to_numpy(epoch_train_loss_r),
    #                  to_numpy(epoch_train_loss_ic),
    #                  to_numpy(epoch_train_loss_bc),
    #                  to_numpy(epoch_train_loss)])

        # scheduler.step()
    # epochs_step2+
    for epoch in range(1, epochs+1):
        # train
        model.train()

        for k, v in model.named_parameters():
            v.requires_grad = True

        epoch_train_loss_r, epoch_train_loss_ic, epoch_train_loss_bc, epoch_train_loss \
            = train_batch(epoch, train_x, train_ic, train_bc, model, optimizer)

        LOSS.append([epoch,
                     to_numpy(epoch_train_loss_r),
                     to_numpy(epoch_train_loss_ic),
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
    # vaild_u = torch.tensor(vaild_u, dtype=torch.float32)

    vaild_x = vaild_x.reshape((100, 100, 2))
    vaild_u = vaild_u.reshape((100, 100, 1))

    vaild_x = vaild_x.to(device)

    u_pred = model(vaild_x)

    # mean l2_error
    l2_error = l2_relative_error(to_numpy(vaild_u), to_numpy(u_pred))
    error_u = abs(vaild_u - to_numpy(u_pred))

    # Save training loss.
    scipy.io.savemat(f'./work/loss/Advection-pinn-mc-loss-wo.mat', {'loss': LOSS})

    # Save predict result
    scipy.io.savemat(f'./work/Advection-pinn-mc-pred-wo.mat', {'x': to_numpy(vaild_x),
                                                           'u_true': vaild_u,
                                                           'u_pred': to_numpy(u_pred),
                                                            'error_u': error_u,
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

    # Exact
    # x = np.linspace(0, 1, 1000)
    # t = np.linspace(0, 0.5, 1000)
    #
    # xx, tt = np.meshgrid(x, t)
    # X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    #
    # u = u(X[..., 0], X[..., 1])

    # X = np.reshape(a, (100, 100, 2))
    # u = np.reshape(b, (100, 100, 1))
    #
    # plt.figure(figsize=(10, 8))
    # plt.pcolormesh(X[..., 1], X[..., 0], u[..., 0], cmap='jet',
    #                shading='gouraud', antialiased=True, snap=True)
    # plt.xlabel("$t$", fontsize=18)
    # plt.ylabel("$x$", fontsize=18)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=18)
    # plt.tight_layout()
    # plt.title("$u(t,x)$", fontsize=18)
    # # plt.savefig('./work/fig/Exact.png', bbox_inches='tight', dpi=600)
    # plt.show()

    ## Pred
    data = scipy.io.loadmat("./work/Advection-pinn-m-pred-wo.mat")
    X, u_true, u_pred, error_u, l2_re = data['x'], data['u_true'], data['u_pred'], data['error_u'], data['l2_re']

    plt.figure(figsize=(10, 8))
    # jet viridis
    plt.pcolormesh(X[..., 1], X[..., 0], error_u[..., 0], cmap='viridis',
                   shading='gouraud', antialiased=True, snap=True)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$x$", fontsize=20)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("$u(t,x)$", fontsize=20)
    # plt.savefig('./work/fig/Error_u_pinn-m.png', bbox_inches='tight', dpi=600)
    plt.show()

    ## loss
    LOSS = scipy.io.loadmat("./work/loss/Advection-pinn-m-loss-wo.mat")['loss']
    epochs, loss_r, loss_ic, loss_bc, loss = LOSS[:, 0], LOSS[:, 1], LOSS[:, 2], LOSS[:, 3], LOSS[:, 4]
    plt.plot(epochs, loss_r, 'bs', markersize=1, label='$loss_r$')
    plt.plot(epochs, loss_ic, 'go', markersize=1, label='$loss_{ic}$')
    plt.plot(epochs, loss_bc, '+', color='purple', markersize=1, label='$loss_{bc}$')
    plt.plot(epochs, loss, 'r^', markersize=1, label='$loss$')
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    plt.ylim(0, 0.0004)
    plt.xlim(1000, 4000)
    plt.axvline(1000, c='black', linestyle='--')
    plt.axvline(2000, c='black', linestyle='--')
    plt.axhline(0.5e-4, c='black', linestyle='--')
    # plt.axvline(4000, c='black', linestyle='--')
    # plt.axvline(8000, c='black', linestyle='--')
    plt.xlabel('Iterations', fontsize=20)
    plt.title("Loss of MultiInNet PINNs", fontsize=13)
    # plt.title("Loss of Multi Step Train MultiInNet PINNs", fontsize=20)
    # plt.title("Loss of Multi Step Train ResNet PINNs")
    # plt.title("Loss of Vanilla PINNs", fontsize=20)
    # plt.title("Loss of Multi Step Train MultiInNet PINNs")
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='best')
    plt.savefig('./work/fig/Adv-pinn-m-loss-wo-ab.png', bbox_inches='tight', dpi=600)
    plt.show()

    print(f'MSE:{((u_pred - u_true)**2).mean()} l2:{l2_relative_error(u_true, u_pred)}')



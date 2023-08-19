#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs Helmholtz 
@File    ：Helmholtz2D.py
@Author  ：LiangL. Yan
"""

"""
    u_xx + u_yy + kappa^2 * u - q(x, y) = 0
    u(-1, y) = u(1, y) = u(x, -1) = u(x, 1) = 0,    x \in [-1, 1], y \in [-1, 1]
    q(x, y) = -(a1 * \pi)^2 * sin(a1 * \pi * x) * sin(a2 * \pi * y) 
              - (a2 * \pi)^2 * sin(a1 * \pi * x) * sin(a2 * \pi * y)
              + kappa^2 * sin(a1 * \pi * x) * sin(a2 * \pi * y) * w
    
    The Exact Analytical Solution:
        u(x,y) = sin(a1 * \pi * x) * sin(a2 * \pi * y)
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


# Basis equations
a_1, a_2, L = 1, 4, 1


def Exact(x, a1, a2, method='numpy'):
    if method == 'numpy':
        return np.sin(a1 * np.pi * x[:, 0:1]) * np.sin(a2 * np.pi * x[:, 1:2])
    else:
        return torch.sin(a1 * torch.pi * x[:, 0:1]) * torch.sin(a2 * torch.pi * x[:, 1:2])


def u_xx(x, a1, a2):
    return - (a1 * torch.pi) ** 2 * torch.sin(a1 * np.pi * x[:, 0:1]) * torch.sin(a2 * torch.pi * x[:, 1:2])


def u_yy(x, a1, a2):
    return - (a2 * torch.pi) ** 2 * torch.sin(a1 * torch.pi * x[:, 0:1]) * torch.sin(a2 * torch.pi * x[:, 1:2])


# Forcing
def f(x, a1, a2, lam=1.0):
    return u_xx(x, a1, a2) + u_yy(x, a1, a2) + lam * Exact(x, a1, a2, method='tensor')


def gen_traindata(N, method='uniform'):

    if method == 'uniform':
        x = np.linspace(-L, L, N, endpoint=False)
        y = np.linspace(-L, L, N, endpoint=False)
        xx, yy = np.meshgrid(x, y)
    elif method == 'sobol':
        a = sobol_sequence.sample(N, 2)
        xx = a[:, 0:1] * 2 * L - L
        yy = a[:, 1:2] * 2 * L - L
    else:
        xx = np.random.random(N) * 2 * L - L
        yy = np.random.random(N) * 2 * L - L

    X = np.vstack((np.ravel(xx), np.ravel(yy))).T
    y = Exact(X, a_1, a_2)

    return X.astype(np.float32), y.astype(np.float32)


def gen_bcdata(N):

    xl, xr = np.linspace(-L, -L, N // 2, endpoint=False), np.linspace(L, L, N // 2, endpoint=False)
    yl, yr = np.linspace(-L, -L, N // 2, endpoint=False), np.linspace(L, L, N // 2, endpoint=False)

    tmp = np.linspace(-L, L, N, endpoint=False)

    x = np.hstack((xl, xr))
    y = np.hstack((yl, yr))
    np.random.shuffle(x)
    np.random.shuffle(y)

    xx1, yy1 = np.meshgrid(x, tmp)
    xx2, yy2 = np.meshgrid(tmp, y)

    X1 = np.vstack((np.ravel(xx1), np.ravel(yy1))).T
    X2 = np.vstack((np.ravel(xx2), np.ravel(yy2))).T
    X = np.vstack((X1, X2))

    return X.astype(np.float32)


def gen_testdata(N):
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    xx, yy = np.meshgrid(x, y)

    X = np.vstack((np.ravel(xx), np.ravel(yy))).T
    y = Exact(X, a_1, a_2)

    return X.astype(np.float32), y.astype(np.float32)


class Helmholtz2D(PINNFree):
    def __init__(self, net):
        super(Helmholtz2D, self).__init__(net)

    def loss_residual(self, x):

        u = self.forward(x)
        du = gradients(x, u)
        d2u = gradients(x, du)

        d2udx2 = d2u[:, 0]
        d2udy2 = d2u[:, 1]

        # U = torch.sin(a_1 * torch.pi * x[:, 0:1]) * torch.sin(a_2 * torch.pi * x[:, 1:2])
        residual = d2udx2 + d2udy2 + u - f(x, a_1, a_2)

        return 1e-5 * (residual**2).mean() + 1 * ((u - Exact(x, a_1, a_2, method='tensor'))**2).mean()

    def loss_ic(self, **kwargs):
        pass

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
    num_train_points = 2000  # double
    train_x, train_u = gen_traindata(num_train_points, method='sobol')
    train_x = torch.tensor(train_x, dtype=torch.float32, requires_grad=True)
    # train_u = torch.tensor(train_u, dtype=torch.float32, requires_grad=True)
    train_x = train_x.to(device)
    # train_u = train_u.to(device)
    # train_x = train_x

    ## boundary condition data
    train_x_bc = gen_bcdata(num_train_points // 2)
    train_x_bc = torch.tensor(train_x_bc, dtype=torch.float32, requires_grad=True)
    train_x_bc = train_x_bc.to(device)

    # Initialize neural network.
    # net = FCNet([2, 50, 50, 50, 1])
    # net = ResNet([2, 20, 20, 20, 20, 1], [20, 20])
    # net = UNet([2, 10, 20, 20, 10, 1], [10, 20])
    # net = MultiInNet([2, 20, 20, 20, 1], [2, 20, 2])
    net = MultiInNetCorrection(2, 1, 20, [20, 20, 20], [20, 20, 20])
    model = Helmholtz2D(net).to(device)
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

    epochs_step1 = 1250
    for epoch in range(1, epochs_step1+1):
        # train
        model.train()
        for k, v in model.named_parameters():
            re_network = re.compile(r'^_basic_net.__OUTPUT__')
            # re_network2 = re.compile(r'^_basic_net.main.6')
            if re_network.match(k):
                # or k == '_basic_net.branch_main.6.weight' or k == '_basic_net.branch_main.6.bias':
                v.requires_grad = True
            else:
                v.requires_grad = False

        epoch_train_loss_r, epoch_train_loss_bc, epoch_train_loss \
            = train_batch(epoch, train_x, train_x_bc, model, optimizer)

        LOSS.append([epoch,
                     to_numpy(epoch_train_loss_r),
                     to_numpy(epoch_train_loss_bc),
                     to_numpy(epoch_train_loss)])

    # scheduler.step()

    epochs_step2 = 1500
    for epoch in range(epochs_step1+1, epochs_step2+1):
        # train
        model.train()
        for k, v in model.named_parameters():
            re_network = re.compile(r'^_basic_net.__OUTPUT__')
            if re_network.match(k):
                v.requires_grad = False
            else:
                v.requires_grad = True

        epoch_train_loss_r, epoch_train_loss_bc, epoch_train_loss \
            = train_batch(epoch, train_x, train_x_bc, model, optimizer)

        LOSS.append([epoch,
                     to_numpy(epoch_train_loss_r),
                     to_numpy(epoch_train_loss_bc),
                     to_numpy(epoch_train_loss)])

    # scheduler.step()

    # epochs_step2 = 0
    for epoch in range(epochs_step2 + 1, epochs + 1):
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
    save_name = 'pinn-mc' + str(epochs) + '.pth'
    model_save_dir = './model/'
    save_model(model_save_dir, save_name, model._basic_net)

    # Evaluate on the whole computational domain
    # Test data
    vaild_x, vaild_u = gen_testdata(100)
    vaild_x = torch.tensor(vaild_x, dtype=torch.float32)

    vaild_x = vaild_x.reshape((100, 100, 2))
    vaild_u = vaild_u.reshape((100, 100, 1))

    vaild_x = vaild_x.to(device)

    u_pred = model(vaild_x)

    # mean l2_error
    l2_error = l2_relative_error(to_numpy(vaild_u), to_numpy(u_pred))
    error_u = abs(vaild_u - to_numpy(u_pred))

    # Save training loss.
    scipy.io.savemat(f'./work/loss/Helmholtz2d-pinn-mc.mat', {'loss': LOSS})

    # Save predict result
    scipy.io.savemat(f'./work/Helmholtz2d-pinn-mc.mat', {'x': to_numpy(vaild_x),
                                                              'u_true': to_numpy(vaild_u),
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

    # Exact
    # X, u = gen_testdata(1000)
    #
    # X = np.reshape(X, (1000, 1000, 2))
    # u = np.reshape(u, (1000, 1000, 1))
    #
    # plt.figure(figsize=(10, 8))
    # plt.pcolormesh(X[..., 1], X[..., 0], u[..., 0], cmap='jet',
    #                shading='gouraud', antialiased=True, snap=True)
    # plt.xlabel("$x$", fontsize=18)
    # plt.ylabel("$y$", fontsize=18)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=18)
    # plt.tight_layout()
    # plt.title("$u(x, y)$", fontsize=18)
    # # plt.savefig('./work/fig/Exact.png', bbox_inches='tight', dpi=600)
    # plt.show()

    # main()

    set_defult_rcParams()

    #################################################
    ###################### Plot #####################
    #################################################

    # Predict
    data = scipy.io.loadmat("./work/Helmholtz2d-pinn-mc.mat")
    X, u_true, u_pred, error_u, l2_re = data['x'], data['u_true'], data['u_pred'], data['error_u'], data['l2_re']

    plt.figure(figsize=(10, 8))
    # jet viridis
    plt.pcolormesh(X[..., 0], X[..., 1], u_true[..., 0], cmap='jet',
                   shading='gouraud', antialiased=True, snap=True)
    plt.xlabel("$x$", fontsize=18)
    plt.ylabel("$y$", fontsize=18)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=18)
    plt.tight_layout()
    # plt.title('$Predict\quadu(x,y)$', fontsize=18)
    plt.title('$Absolute\quadError$', fontsize=18)
    # plt.savefig('./work/fig/u_pinn-mc.png', bbox_inches='tight', dpi=600)
    plt.show()

    ## Loss
    LOSS = scipy.io.loadmat("./work/loss/Helmholtz2d-pinn-n.mat")['loss']
    epochs, loss_r, loss_bc, loss = LOSS[:, 0], LOSS[:, 1], LOSS[:, 2], LOSS[:, 3]
    plt.plot(epochs, loss_r, 'bs', markersize=1, label='$loss_r$')
    plt.plot(epochs, loss_bc, '+', color='purple', markersize=1, label='$loss_{bc}$')
    plt.plot(epochs, loss, 'r^', markersize=1, label='$loss$')
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    plt.ylim(0, 0.35)
    # plt.xlim(0, 4000)
    plt.axhline(1e-1, c='black', linestyle='--')
    # plt.axvline(1000, c='black', linestyle='--')
    plt.axvline(2000, c='black', linestyle='--')
    # plt.axvline(4000, c='black', linestyle='--')
    # plt.axvline(8000, c='black', linestyle='--')
    plt.xlabel('Iterations', fontsize=20)
    # plt.title("Loss of Multi Step Train MultiInNet PINNs")
    plt.title("Loss of Multi Step Train PINNs", fontsize=16)
    # plt.title("Loss of Multi Step Train MultiInNet PINNs")
    plt.legend(loc='best')
    # plt.savefig('./work/fig/Helmholtz2d-pinn-loss-n.png', bbox_inches='tight', dpi=600)
    plt.show()

    print(f'MSE:{((u_pred - u_true) ** 2).mean()} l2:{l2_relative_error(u_true, u_pred)}')

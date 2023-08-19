#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs Burger
@File    ：Burger.py
@Author  ：LiangL. Yan
"""


import sys
sys.path.append("../..")
from SALib.sample import sobol_sequence
import numpy as np
from src.models.Base import PINN
import torch
from src.utils.operator import gradients, to_numpy, l2_relative_error
import time
from src.utils.utils import save_model
import scipy.io
import os
import matplotlib.pyplot as plt


# Seeds
# torch.manual_seed(123456)
# np.random.seed(123456)


def gen_testdata():

    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X.astype(np.float32), y.astype(np.float32)


def gen_traindata(N, method='uniform'):

    if method == 'uniform':
        Nt = int((N / 2) ** 0.5)
        Nx = 2 * Nt
        x = np.linspace(-1, 1, Nx, endpoint=False)
        t = np.linspace(0, 1, Nt, endpoint=False)
        xx, tt = np.meshgrid(x, t)
    elif method == 'sobol':
        # n = int(N*0.05)
        a = sobol_sequence.sample(N, 2)
        xx = a[:, 0:1] * 2 - 1
        tt = a[:, 1:2]
    else:
        # n = int(N * 0.05)
        xx = np.random.random(N) * 2 - 1
        tt = np.random.random(N)

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    return X.astype(np.float32)


def gen_initdata(N):
    """u(0, x) = -sin(Πx)."""
    Nt = int((N / 2) ** 0.5)
    Nx = 2 * Nt
    x = np.linspace(-1, 1, Nx, endpoint=False)
    t = np.linspace(0, 0, Nt, endpoint=False)
    xx, tt = np.meshgrid(x, t)

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    y = []
    for item in X:
        y.append(-np.sin(np.pi * item[0]))

    y = np.reshape(y, (X.shape[0], -1))

    return X.astype(np.float32), y.astype(np.float32)


def gen_bcdata(N):
    """u(t, -1) = u(t, 1) = 0"""
    Nt = int((N / 2) ** 0.5)
    Nx = 2 * Nt
    xl = np.linspace(-1, -1, Nx, endpoint=False)
    xr = np.linspace(1, 1, Nx, endpoint=False)
    x = np.hstack((xl, xr))
    np.random.shuffle(x)

    t = np.linspace(0, 1, Nt, endpoint=False)

    xx, tt = np.meshgrid(x, t)

    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    return X.astype(np.float32)


class Burger(PINN):
    def __init__(self, in_=2, out=1, d=9, active=torch.nn.ReLU()):
        super(Burger, self).__init__(input_dim=in_, output_dim=out, Netdeep=d, active=active)

    def output_transform(self, x, y):
        x_in = x[:, 0:1]
        t_in = x[:, 1:2]

        return (1 - x_in) * (1 + x_in) * (1 - torch.exp(-t_in)) * y - torch.sin(np.pi * x_in)

    def loss_residual(self, x):
        """PDE loss."""
        y = self.forward(x)

        dy = gradients(x, y)
        d2y = gradients(x, dy)

        dydx = dy[:, 0:1]
        dydt = dy[:, 1:2]
        d2ydx2 = d2y[:, 0:1]

        eqs = dydt + y * dydx - (0.01 / np.pi) * d2ydx2

        return (eqs ** 2).mean()

    def loss_ic(self, x, y_true):
        y = self.forward(x)
        return ((y - y_true) ** 2).mean()

    def loss_bc(self, x):
        y = self.forward(x)
        return (y ** 2).mean()


def train_batch(epoch, train_data, train_x_ic, train_ic_u, train_x_bc, model, optimizer):
    """Train a model in one epoch."""
    LOSS = []

    def closure():
        optimizer.zero_grad()
        # compute loss
        loss_r = model.loss_residual(train_data)
        loss_ic = model.loss_ic(train_x_ic, train_ic_u)
        loss_bc = model.loss_bc(train_x_bc)

        w_r, w_u = 1, 5
        loss = w_r * loss_r + w_u * loss_ic + w_u * loss_bc

        LOSS.append(loss_r)
        LOSS.append(loss_ic)
        LOSS.append(loss_bc)
        LOSS.append(loss)

        print(f'epoch {epoch} loss_pde:{loss_r:.2e}, loss_ic:{loss_ic:.2e}, loss_bc:{loss_bc:.2e}')
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
    # residual
    num_train_points = 1000
    train_x = gen_traindata(num_train_points, method='sobol')
    train_x = torch.tensor(train_x, dtype=torch.float32, requires_grad=True)
    train_x = train_x.to(device)

    # init condition data
    train_x_ic, train_ic_u = gen_initdata(100)
    train_x_ic = torch.tensor(train_x_ic, dtype=torch.float32, requires_grad=True)
    train_ic_u = torch.tensor(train_ic_u, dtype=torch.float32, requires_grad=True)
    train_x_ic, train_ic_u = train_x_ic.to(device), train_ic_u.to(device)

    # boundary condition data
    train_x_bc = gen_bcdata(100)
    train_x_bc = torch.tensor(train_x_bc, dtype=torch.float32, requires_grad=True)
    train_x_bc = train_x_bc.to(device)

    # Initialize neural network.
    model = Burger().to(device)
    # print(model)

    # training parameters.
    epochs, lr = 4000, 1e-3

    # optimizer.
    optimizer0 = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer1 = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=5)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer0, step_size=2000, gamma=0.90)

    # loss log.
    LOSS = []

    # start time
    print("## Start Training...")
    tic = time.time()

    # Simple Train.
    for epoch in range(1, epochs+1):
        # train
        model.train()
        epoch_train_loss_r, epoch_train_loss_ic, epoch_train_loss_bc, epoch_train_loss \
            = train_batch(epoch, train_x, train_x_ic, train_ic_u, train_x_bc, model, optimizer0)

        # model.eval()
        # epoch_test_loss_r, epoch_test_loss_ic, epoch_test_loss = test(x_test, model)

        LOSS.append([epoch,
                     to_numpy(epoch_train_loss_r),
                     to_numpy(epoch_train_loss_ic),
                     to_numpy(epoch_train_loss_bc),
                     to_numpy(epoch_train_loss)])

        scheduler.step()

    # for epoch in range(3000, epochs+1):
    #     # train
    #     model.train()
    #     epoch_train_loss_r, epoch_train_loss_ic, epoch_train_loss_bc, epoch_train_loss \
    #         = train_batch(epoch, train_x, train_x_ic, train_ic_u, train_x_bc, model, optimizer1)
    #
    #     # model.eval()
    #     # epoch_test_loss_r, epoch_test_loss_ic, epoch_test_loss = test(x_test, model)
    #
    #     LOSS.append([epoch,
    #                  to_numpy(epoch_train_loss_r),
    #                  to_numpy(epoch_train_loss_ic),
    #                  to_numpy(epoch_train_loss_bc),
    #                  to_numpy(epoch_train_loss)])

    toc = time.time()
    print(f'Total training time: {toc - tic}')
    print('\nTrain Done......')

    # Save the network parameters
    save_name = 'pinn-' + str(epochs) + '.pth'
    model_save_dir = './model/'
    save_model(model_save_dir, save_name, model._basic_net)

    # Evaluate on the whole computational domain
    # Test data
    valid_x, valid_u = gen_testdata()
    x_true = torch.tensor(valid_x.reshape((100, 256, 2)), dtype=torch.float32, requires_grad=True)
    u_true = valid_u.reshape((100, 256, 1))

    x_true = x_true.to(device)
    u_pred = model(x_true)
    # u_pred = model.output_transform(x_true, u_pred)

    # ERROR
    error_u = abs(u_true - to_numpy(u_pred))
    # error_pde = abs(to_numpy(model.loss_residual(x_true)))
    # l2_re = l2_relative_error(u_true, to_numpy(u_pred))

    # Save training loss.
    scipy.io.savemat(f'./work/loss/Burger-pinn-loss-relu.mat', {'loss': LOSS})

    # Save predict result
    scipy.io.savemat(f'./work/Burger-pinn-pred-relu.mat', {'x': to_numpy(x_true),
                                                  'u_true': u_true,
                                                  'u_pred': to_numpy(u_pred),
                                                  'error_u': error_u,
                                                  # 'error_pde': error_pde,
                                                  # 'l2_re': l2_re
                                                      }

                     )


if __name__ == "__main__":

    # Create directories if not exist.
    log_dir = './work/loss/'
    model_save_dir = "./model/"

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # main()

    # Plot
    data = scipy.io.loadmat('work/Burger-pinn-pred-relu.mat')
    x_true, u_true, u_pred = data['x'], data['u_true'], data['u_pred']
    error_u = data['error_u']

    # u_pred
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(x_true[..., 1], x_true[..., 0], u_pred[..., 0], cmap='jet',
                   shading='gouraud', antialiased=True, snap=True)
    plt.xlabel("$t$", fontsize=18)
    plt.ylabel("$x$", fontsize=18)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.title("$u(t,x)$")
    # plt.savefig('./work/fig/pinn-pred-4000.png', bbox_inches='tight', dpi=600)
    plt.show()

    # error_u
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(x_true[..., 1], x_true[..., 0], error_u[..., 0], cmap='viridis',
                   shading='gouraud', antialiased=True, snap=True
                   )
    plt.xlabel("$t$", fontsize=18)
    plt.ylabel("$x$", fontsize=18)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.title("Error of $u$ with ReLU")
    # plt.savefig("./work/fig/error-u-pinn-adam+lbfgs.png", bbox_inches='tight', dpi=600)
    plt.show()

    # loss
    LOSS = scipy.io.loadmat('./work/loss/Burger-pinn-loss-relu.mat')
    epochs, loss_r, loss_ic, loss_bc, loss = LOSS['loss'][:, 0], \
                                             LOSS['loss'][:, 1], LOSS['loss'][:, 2], LOSS['loss'][:, 3], \
                                             LOSS['loss'][:, 4]
    plt.plot(epochs, loss, 'r^', markersize=1, label='$loss$')
    plt.plot(epochs, loss_r, 'bs', markersize=1, label='$loss_r$')
    plt.plot(epochs, loss_ic, 'go', markersize=1, label='$loss_{ic}$')
    plt.plot(epochs, loss_bc, '+', color='purple', markersize=1, label='$loss_{bc}$')
    plt.title('Loss of PINN with ReLU')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.ylim(0, 0.45)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    plt.savefig("./work/fig/Burger-pinn-loss-relu.png", bbox_inches='tight', dpi=600)
    plt.show()



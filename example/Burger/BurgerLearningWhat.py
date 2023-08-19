#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs BurgerLearningWhat
@File    ：BurgerLearningWhat.py
@Author  ：LiangL. Yan
"""
import sys

sys.path.append("../..")
from SALib.sample import sobol_sequence
import numpy as np
from src.models.Base import PINNFree
from src.models.NN import Network
import torch
import matplotlib.pyplot as plt
from src.utils.operator import gradients, to_numpy
import time
from src.utils.utils import save_model
import scipy.io
import os
from src.utils.visualizer import set_defult_rcParams

# Seeds
torch.manual_seed(123456)
np.random.seed(123456)


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


class Burger(PINNFree):
    def __init__(self, net):
        super(Burger, self).__init__(net=net)

    def loss_residual(self, x):
        """PDE loss."""
        _, _, y = self.forward(x)

        dy = gradients(x, y)
        d2y = gradients(x, dy)

        dydx = dy[:, 0:1]
        dydt = dy[:, 1:2]
        d2ydx2 = d2y[:, 0:1]

        eqs = dydt + y * dydx - (0.01 / np.pi) * d2ydx2

        return (eqs ** 2).mean()

    def loss_ic(self, x, y_true):
        _, _, y = self.forward(x)
        return ((y - y_true) ** 2).mean()

    def loss_bc(self, x):
        _, _, y = self.forward(x)
        return (y ** 2).mean()


def train_batch(epoch, train_data, train_x_ic, train_ic_u, train_x_bc, model, optimizer, W):
    """Train a model in one epoch."""
    LOSS = []

    def closure():
        optimizer.zero_grad()
        # compute loss
        ## loss_pde
        loss_r = model.loss_residual(train_data)
        ## loss_ic
        loss_ic = model.loss_ic(train_x_ic, train_ic_u)
        ## loss_bc
        loss_bc = model.loss_bc(train_x_bc)

        w_r, w_ic, w_bc = W[0], W[1], W[2]
        loss = w_r * loss_r + w_ic * loss_ic + w_bc * loss_bc

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

    # All of conditions.
    # valid_x, valid_u = gen_testdata()
    # valid_x = torch.tensor(valid_x, dtype=torch.float32, requires_grad=True)
    # valid_u = torch.tensor(valid_u, dtype=torch.float32, requires_grad=True)
    # valid_x, valid_u = valid_x.to(device), valid_u.to(device)

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
    net = Network([2, 20, 20, 20, 20, 1]).to(device)
    model = Burger(net).to(device)
    # print(model)

    # training parameters.
    epochs, lr = 8000, 1e-3

    # optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.90)

    # loss log.
    LOSS = []

    # start time
    print("## Start Training...")
    tic = time.time()

    # Simple Train.
    for epoch in range(1, epochs+1):
        # train
        model.train()
        # train step 3: pretraining whole network.
        for k, v in model.named_parameters():
            v.requires_grad = True

        W = [1, 1, 1]
        epoch_train_loss_r, epoch_train_loss_ic, epoch_train_loss_bc, epoch_train_loss \
            = train_batch(epoch, train_x, train_x_ic, train_ic_u, train_x_bc, model, optimizer, W)

        LOSS.append([epoch,
                     to_numpy(epoch_train_loss_r),
                     to_numpy(epoch_train_loss_ic),
                     to_numpy(epoch_train_loss_bc),
                     to_numpy(epoch_train_loss)])

        scheduler.step()

    toc = time.time()
    print(f'Total training time: {toc - tic}')
    print('\nTrain Done......')

    # Save training loss.
    scipy.io.savemat(f'./work/loss/Burger-pinn-loss_l.mat', {'loss': LOSS})

    # Save the network parameters
    save_name = 'pinn-l' + str(epochs) + '.pth'
    model_save_dir = './model/'
    save_model(model_save_dir, save_name, model._basic_net)

    # Evaluate on the whole computational domain
    # Test data
    valid_x, valid_u = gen_testdata()
    x_true = torch.tensor(valid_x.reshape((100, 256, 2)), dtype=torch.float32, requires_grad=True)
    u_true = valid_u.reshape((100, 256, 1))

    x_true = x_true.to(device)
    # _, _, u_pred = model(x_true)
    input_pred, hidden_pred, u_pred = model(x_true)

    input_pred = to_numpy(input_pred)
    hidden_pred = to_numpy(hidden_pred)
    u_pred = to_numpy(u_pred)
    x_true = to_numpy(x_true)

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    # input_pred
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(x_true[..., 1], x_true[..., 0], input_pred[..., 0], cmap='jet',
                   shading='gouraud', antialiased=True, snap=True)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$x$", fontsize=20)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("$input(t,x)$", fontsize=20)
    plt.savefig('./work/fig/pinn-input-pred.png', bbox_inches='tight', dpi=600)
    plt.show()

    # hidden_pred
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(x_true[..., 1], x_true[..., 0], hidden_pred[..., 0], cmap='jet',
                   shading='gouraud', antialiased=True, snap=True)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$x$", fontsize=20)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("$hidden(t,x)$", fontsize=20)
    plt.savefig('./work/fig/pinn-hidden-pred.png', bbox_inches='tight', dpi=600)
    plt.show()

    # u_pred
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(x_true[..., 1], x_true[..., 0], u_pred[..., 0], cmap='jet',
                   shading='gouraud', antialiased=True, snap=True)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("$x$", fontsize=20)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.title("$u(t,x)$", fontsize=20)
    plt.savefig('./work/fig/pinn-pred.png', bbox_inches='tight', dpi=600)
    plt.show()

    # Plot loss
    LOSS = scipy.io.loadmat('./work/loss/Burger-pinn-loss_l.mat')
    epochs, loss_r, loss_ic, loss_bc, loss = LOSS['loss'][:, 0], \
                                             LOSS['loss'][:, 1], LOSS['loss'][:, 2], LOSS['loss'][:, 3], \
                                             LOSS['loss'][:, 4]
    plt.plot(epochs, loss, 'r^', markersize=1, label='$loss$')
    plt.plot(epochs, loss_r, 'bs', markersize=1, label='$loss_r$')
    plt.plot(epochs, loss_ic, 'go', markersize=1, label='$loss_{ic}$')
    plt.plot(epochs, loss_bc, '+', color='purple', markersize=1, label='$loss_{bc}$')
    # plt.axvline(3000, c='black', linestyle='--')
    # plt.axvline(3500, c='black', linestyle='--')
    # plt.axvline(3750, c='black', linestyle='--')
    plt.xlabel('Iterations', fontsize=20)
    plt.title("Loss of Vanilla PINN", fontsize=20)
    plt.legend(loc='best')
    plt.ylim(0, 0.2)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    plt.savefig("./work/fig/Burger-pinn-loss.png", bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == "__main__":

    # Create directories if not exist.
    log_dir = './work/loss/'
    model_save_dir = "./model/"

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    set_defult_rcParams()

    main()

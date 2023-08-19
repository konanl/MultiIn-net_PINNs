#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs SchrodingerLearningWhat
@File    ：SchrodingerLearningWhat.py
@Author  ：LiangL. Yan
"""


import sys

import winnt

sys.path.append("../..")

from SALib.sample import sobol_sequence
import numpy as np
from src.models.Base import PINN, PINNFree
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.operator import gradients, to_numpy, l2_relative_error
from src.utils.visualizer import newfig
from src.models.NN import Network
import time
from src.utils.utils import save_model
import scipy.io
import os
from pyDOE import lhs
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import math
from src.utils.visualizer import set_defult_rcParams, set_bwith


# Seeds
torch.manual_seed(123456)
np.random.seed(123456)


set_defult_rcParams()


class SchrodingerBoundary(Dataset):
    def __init__(self):
        samples = 50
        samples_half = math.floor(samples/2)
        self.t = torch.tensor(np.squeeze(lhs(1, samples=samples_half) * np.pi/2), requires_grad=True).float()
        return

    def __getitem__(self, idx):
        return self.t[idx]

    def __len__(self):
        return len(self.t)

    def getall(self):
        return self.t


class SchrodingerInitial(Dataset):
    def __init__(self):
        h_func = lambda x : 2 * (1/torch.cosh(x))
        samples = 50
        self.x = torch.tensor((lhs(1, samples=samples)*10 - 5), requires_grad=True).float()
        self.h = torch.squeeze(torch.stack((
            torch.tensor(h_func(self.x)).float(),
            torch.zeros((len(self.x), 1))
        ), 1))
        return

    def __getitem__(self, idx):
        return self.x[idx], self.h[idx]

    def __len__(self):
        return len(self.x)

    def getall(self):
        return self.x, self.h


class SchrodingerF(Dataset):
    def __init__(self): # returns x,t
        samples = 20000
        x = torch.tensor(lhs(1, samples=samples)*10 - 5, requires_grad=True).float()
        t = torch.tensor(lhs(1, samples=samples)*(np.pi/2), requires_grad=True).float()
        self.X = torch.squeeze(torch.dstack((x, t))).float()
        return

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)

    def getall(self):
        return self.X


class Schrodinger(PINNFree):
    def __init__(self, net):
        super(Schrodinger, self).__init__(net=net)

    def h(self, x, t):
        X = torch.cat((x, t), 1).float()
        return self.forward(X)

    def norm(self, r, c):
        return torch.sqrt((r**2) + (c**2))

    def loss_residual(self, x, t):

        _, _, y = self.h(x, t)

        u = torch.unsqueeze(y[:, 0], 1)
        v = torch.unsqueeze(y[:, 1], 1)

        dudt = gradients(t, u)
        dvdt = gradients(t, v)

        dudx = gradients(x, u)
        dudv = gradients(x, v)

        d2ux2 = gradients(x, dudx)
        d2vx2 = gradients(x, dudv)

        f_u = dudt + 0.5 * d2vx2 + (u**2 + v**2) * v
        f_v = dvdt - 0.5 * d2ux2 - (u**2 + v**2) * u

        return torch.mean(self.norm(f_u, f_v)**2)

    def loss_ic(self, X, h_true):
        X = torch.concat((X[0], X[1]), 1).float()
        _, _, h_pred = self.forward(X)

        return torch.mean(torch.square(h_pred - h_true))

    def loss_bc(self, x, t):

        _, _, y_lb = self.h(x, t)
        _, _, y_ub = self.h(-x, t)

        ul, uu = y_lb[:, 0], y_ub[:, 0]
        vl, vu = y_lb[:, 1], y_ub[:, 1]

        du_lb_pred = gradients(x, ul)
        du_ub_pred = gradients(x, uu)
        dv_lb_pred = gradients(x, vl)
        dv_ub_pred = gradients(x, vu)

        h_u_err = torch.unsqueeze(ul - uu, 1)
        h_v_err = torch.unsqueeze(vl - vu, 1)

        dhdu_err = du_lb_pred - du_ub_pred
        dhdv_err = dv_lb_pred - dv_ub_pred

        h_err = (h_u_err**2) + (h_v_err**2)
        dh_err = (dhdu_err**2) + (dhdv_err**2)

        return torch.mean((h_err**2) + (dh_err**2))


def train_batch(epoch, train_data, train_ic, train_bc, model, optimizer, W=[1, 1, 1]):
    LOSS = []

    def closure():
        optimizer.zero_grad()
        # compute loss
        loss_r = model.loss_residual(train_data[0], train_data[1])
        loss_ic = model.loss_ic(train_ic[:-1], train_ic[2])
        loss_bc = model.loss_bc(train_bc[0], train_bc[1])

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
    boundary_ds = SchrodingerBoundary()
    initial_ds = SchrodingerInitial()
    schrodinger_ds = SchrodingerF()
    schrodinger_dl = DataLoader(schrodinger_ds, batch_size=4096, shuffle=True)

    x0, h0 = initial_ds.getall()
    t0 = torch.zeros((len(x0), 1)).float()

    tb = torch.unsqueeze(boundary_ds.getall(), 1)
    xb = torch.ones((len(tb), 1), requires_grad=True) * 5.

    X = schrodinger_ds.getall()
    x = torch.unsqueeze(X[:, 0], 1)
    t = torch.unsqueeze(X[:, 1], 1)

    x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    t0 = torch.tensor(t0, dtype=torch.float32, requires_grad=True)
    h0 = torch.tensor(h0, dtype=torch.float32, requires_grad=True)

    xb = torch.tensor(xb, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(tb, dtype=torch.float32, requires_grad=True)

    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)

    x0, t0, h0, xb, tb, x, t = x0.to(device), t0.to(device), h0.to(device), \
                                  xb.to(device), tb.to(device), x.to(device), t.to(device)

    # model
    net = Network([2, 100, 100, 100, 100, 2]).to(device)
    model = Schrodinger(net).to(device)
    # print(model)

    # training parameters.
    epochs, lr = 12000, 1e-3

    # optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer0 = torch.optim.LBFGS(model.parameters(), lr=lr*10,
                                  max_iter=100, max_eval=None, tolerance_grad=1e-15,
                                  tolerance_change=1e-09, history_size=500, line_search_fn="strong_wolfe")

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.90)

    # loss log.
    LOSS = []

    # Start time.
    print("## Start Training...")
    tic = time.time()

    for epoch in range(0, epochs+1):
        # train
        model.train()

        for k, v in model.named_parameters():
            v.requires_grad = True

        epoch_loss_r, epoch_loss_ic, epoch_loss_bc, \
        epoch_loss = train_batch(epoch, [x, t], [x0, t0, h0], [xb, tb], model, optimizer)

        LOSS.append([epoch,
                     to_numpy(epoch_loss_r),
                     to_numpy(epoch_loss_ic),
                     to_numpy(epoch_loss_bc), to_numpy(epoch_loss)])

        scheduler.step()

    toc = time.time()

    scipy.io.savemat(f'./work/loss/Schrodinger-pinn-loss.mat', {'loss': LOSS})

    print(f'Total training time: {toc - tic}')
    print('\nTrain Done......')

    # Save the network parameters
    save_name = 'pinn-l' + str(epochs) + '.pth'
    model_save_dir = './model/'
    save_model(model_save_dir, save_name, model._basic_net)

    # Evaluate on the whole computational domain
    xx = np.linspace(-5, 5, 300)
    tt = np.linspace(0, np.pi / 2, 300)

    X, T = np.meshgrid(xx, tt)

    _X = torch.tensor(np.dstack((X, T))).float().to(device)

    input_h_hat, hidden_h_hat, h_hat = model.forward(_X)

    input_h_hat = to_numpy(input_h_hat)
    h_hat = to_numpy(h_hat)
    hidden_h_hat = to_numpy(hidden_h_hat)

    input_h_hat_norm = np.sqrt((input_h_hat[:, :, 0] ** 2) + (input_h_hat[:, :, 1] ** 2))
    h_norm = np.sqrt((h_hat[:, :, 0] ** 2) + (h_hat[:, :, 1] ** 2))
    hidden_h_hat_norm = np.sqrt((hidden_h_hat[:, :, 0] ** 2) + (hidden_h_hat[:, :, 1] ** 2))

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    # h
    fig = plt.figure(figsize=(10, 4))
    ax = plt.axes()

    cf = ax.contourf(T, X, h_norm, 30, cmap='YlGnBu')
    plt.colorbar(cf)

    plt.title('$|h(x,t)|$', fontsize=20)
    plt.ylabel("$Position (x)$", fontsize=20)
    plt.xlabel("$Time (t)$", fontsize=20)
    plt.savefig("./work/fig/h_pred_pinn.png", bbox_inches='tight', dpi=600)
    plt.show()

    # input
    fig = plt.figure(figsize=(10, 4))
    ax = plt.axes()

    cf = ax.contourf(T, X, input_h_hat_norm, 30, cmap='YlGnBu')
    plt.colorbar(cf)

    plt.title('$|h(x,t)|$', fontsize=20)
    plt.ylabel("$Position (x)$", fontsize=20)
    plt.xlabel("$Time (t)$", fontsize=20)
    plt.savefig("./work/fig/input_h_pred_pinn.png", bbox_inches='tight', dpi=600)
    plt.show()

    # input
    fig = plt.figure(figsize=(10, 4))
    ax = plt.axes()

    cf = ax.contourf(T, X, hidden_h_hat_norm, 30, cmap='YlGnBu')
    plt.colorbar(cf)

    plt.title('$|h(x,t)|$', fontsize=20)
    plt.ylabel("$Position (x)$", fontsize=20)
    plt.xlabel("$Time (t)$", fontsize=20)
    plt.savefig("./work/fig/hidden_h_pred_pinn.png", bbox_inches='tight', dpi=600)
    plt.show()

    # loss
    LOSS = scipy.io.loadmat('./work/loss/Schrodinger-pinn-loss.mat')
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
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.ylim(0, 0.1)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    # plt.savefig("./work/fig/Sch-pinn-loss.png", bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':

    # Create directories if not exist.
    log_dir = './work/loss/'
    model_save_dir = "./model/"

    set_defult_rcParams()
    # set_bwith()

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    main()

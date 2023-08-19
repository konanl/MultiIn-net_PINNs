'''
Descripttion: 
version: 
Author: Yoking.L
Date: 2023-08-19 14:31:51
LastEditTime: 2023-08-19 14:36:09
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs / utils.visualizer 
@File    ：visualizer.py
@Author  ：LiangL. Yan
"""

import matplotlib.pyplot as plt
import numpy as np


def set_defult_rcParams():
    """Set the style of figure."""
    params = {
        "font.family": "serif",
        "font.size": 18.0,
        "legend.handlelength": 0.5,
        # "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,

        "axes.linewidth": 1.5,
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.major.size": 4,
        "ytick.minor.size": 2,

        "xtick.labelsize": 'x-small',
        "ytick.labelsize": 'x-small',
        "axes.labelsize": 'small',
    }
    plt.rcParams.update(params)


def set_bwith(bwith=1.5):
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

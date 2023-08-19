'''
Descripttion: 
version: 
Author: Yoking.L
Date: 2023-08-19 14:31:51
LastEditTime: 2023-08-19 14:36:45
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MultiIn_net PINNs / utils.Sampler
@File    ：Sampler.py
@Author  ：LiangL. Yan
"""
import numpy as np
from SALib.sample import sobol_sequence


# see:
## https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py
class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y

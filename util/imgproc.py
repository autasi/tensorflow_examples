#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np


# global contrast normalization
#def GCN(x, scale = 1.0, lmbd = 0.0, epsilon = 10e-8):
#    xgcn = np.zeros_like(x)
#    for n in range(x.shape[0]):
#        xn = x[n] - x[n].mean()
#        contrast = np.sqrt(lmbd + np.mean(xn**2))
#        xgcn[n] = scale * xn / contrast.clip(epsilon)
#    return xgcn

def GCN(x, scale = 1.0, lmbd = 0.0, epsilon = 10e-6):
    shape = x.shape
    dims = np.prod(shape[1:])
    x = x.reshape((shape[0], dims))
    mean = x.mean(axis=1)
    x = x - mean[:, np.newaxis]
    contrast = np.sqrt(lmbd + (x ** 2).sum(axis=1))
    contrast = contrast.clip(epsilon)
    x /= contrast[:, np.newaxis]
    x *= scale
    x = x.reshape(shape)
    return x
    
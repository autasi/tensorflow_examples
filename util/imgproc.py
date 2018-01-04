#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

def ZCA_whitening(X, is_centered = False, eps = 1e-5):
    shape = X.shape
    dims = np.prod(shape[1:])
    X = X.reshape((shape[0], dims))
    if not is_centered:
        mean = X.mean(axis = 0)
        X = X - mean
    cov = np.cov(X, rowvar=True)
    U,S,V = np.linalg.svd(cov)
    s = np.sqrt(S.clip(eps))
    zca_matrix = np.dot(U, np.dot(np.diag(1.0/s), U.T))
    zca = np.dot(zca_matrix, X) 
    return zca.reshape(shape)

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def global_mean(tr, te = None):
    tr_mean = np.mean(tr, axis = (0,1,2,3))
    tr = tr-tr_mean
    if te is not None:
        te = te-tr_mean
        return (tr, te)
    else:
        return tr


def global_mean_std(tr, te = None, eps = 1e-7):
    tr_mean = np.mean(tr, axis = (0,1,2,3))
    tr_std = np.std(tr, axis = (0,1,2,3))
    tr = (tr-tr_mean)/(tr_std+eps)
    if te is not None:
        te = (te-tr_mean)/(tr_std+eps)
        return (tr, te)
    else:
        return tr


def channel_mean(tr, te = None):    
    tr_mean = np.mean(tr, axis = (0,1,2))
    tr = tr-tr_mean
    if te is not None:
        te = te-tr_mean
        return (tr, te)
    else:
        return tr

    
def channel_mean_std(tr, te = None, eps = 1e-7):    
    tr_mean = np.mean(tr, axis = (0,1,2))
    tr_std = np.std(tr, axis = (0,1,2))
    tr = (tr-tr_mean)/(tr_std+eps)
    if te is not None:
        te = (te-tr_mean)/(tr_std+eps)
        return (tr, te)
    else:
        return tr
    
    
def pixel_mean(tr, te = None):    
    tr_mean = np.mean(tr, axis = 0)
    tr = tr-tr_mean
    if te is not None:
        te = te-tr_mean
        return (tr, te)
    else:
        return tr

    
def pixel_mean_std(tr, te = None, eps = 1e-7):    
    tr_mean = np.mean(tr, axis = 0)
    tr_std = np.std(tr, axis = 0)
    tr = (tr-tr_mean)/(tr_std+eps)
    if te is not None:
        te = (te-tr_mean)/(tr_std+eps)
        return (tr, te)
    else:
        return tr

#https://www.kaggle.com/c/cifar-10/discussion/6318
#https://gist.github.com/kastnerkyle/9822570
#https://github.com/kastnerkyle/kaggle-cifar10/blob/master/kaggle_train.py
#https://github.com/nagadomi/kaggle-cifar10-torch7
def ZCA_whitening(tr, te = None, is_centered = False, eps = 1e-5):
    tr_shape = tr.shape
    dims = np.prod(tr_shape[1:])
    tr = tr.reshape((tr_shape[0], dims))
    if not is_centered:
        tr_mean = tr.mean(axis = 0)
        tr = tr - tr_mean
    cov = np.cov(tr, rowvar=False)
    U,S,V = np.linalg.svd(cov)
    s = np.sqrt(S.clip(eps))
    zca_matrix = np.dot(U, np.dot(np.diag(1.0/s), U.T))
    tr_zca = np.dot(tr, zca_matrix) 
    tr = tr_zca.reshape(tr_shape)
    if te is not None:
        te_shape = te.shape
        te = te.reshape((te_shape[0], dims))
        te = te - tr_mean
        te_zca = np.dot(te, zca_matrix)
        te = te_zca.reshape(te_shape)
        return (tr, te)
    else:
        return tr
    
    
def ZCA_whitening2(tr, te = None, is_centered = False, eps = 1e-5):
    tr_shape = tr.shape
    dims = np.prod(tr_shape[1:])
    tr = tr.reshape((tr_shape[0], dims))
    if not is_centered:
        tr_mean = tr.mean(axis = 0)
        tr -= tr_mean
    U, S, VT = sp.linalg.svd(tr, full_matrices=False)
    S = S.clip(eps)
    zca_matrix = np.dot(VT.T * np.sqrt(1.0 / (S ** 2)), VT)
    zca_matrix = zca_matrix.T
    tr = np.dot(tr, zca_matrix)
    tr = tr.reshape(tr_shape)
    if te is not None:
        te_shape = te.shape
        te = te.reshape((te_shape[0], dims))
        if not is_centered:
            te = te - tr_mean
        te = np.dot(te, zca_matrix)
        te = te.reshape(te_shape)
        return (tr, te)
    else:
        return tr


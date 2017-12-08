#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def batch_generator(size, x, y=None, fixed_size=True):
    """Generates batches from one or two input arrays.
    Args:
        size: An integer representing the batch size.
        x: An ndarray representing the input array.
        y: An optional ndarray representing another input array.
        fixed_size: A boolean that defines how batches are generated. If True,
            then all batches have fixed size. If False, then the last batch is
            allowed to have a smaller size.
    Yields:
        ndarray or tuple of ndarrays: If `y` is None, then a single ndarray
            is generated, otherwise a tuple of two ndarrays.
    """
    ndata = x.shape[0]
    if fixed_size:
        nbatch = ndata // size
        for b in range(0, nbatch):
            bfrom = b*size
            bto = bfrom+size
            if y is not None:
                yield (x[bfrom:bto], y[bfrom:bto])
            else:
                yield x[bfrom:bto]
        if ndata % size > 0:            
            bfrom = min(size, ndata) # take data from the end
            if y is not None:
                yield (x[-bfrom:], y[-bfrom:])
            else:
                yield x[-bfrom:]
    else:
        nbatch = (ndata + size - 1) // size
        for b in range(0, nbatch):
            bfrom = b*size
            bto = min(bfrom+size, ndata)
            if y is not None:
                yield (x[bfrom:bto], y[bfrom:bto])
            else:
                yield x[bfrom:bto]
                
                
def random_batch_generator(size, x, y=None, fixed_size=True, seed=None):            
    """Generates randomized batches from one or two input arrays.
    Args:
        size: An integer representing the batch size.
        x: An ndarray representing the input array.
        y: An optional ndarray representing another input array.
        fixed_size: A boolean that defines how batches are generated. If True,
            then all batches have fixed size. If False, then the last batch is
            allowed to have a smaller size.
        seed: An optional integer representing the random seed number.
    Yields:
        ndarray or tuple of ndarrays: If `y` is None, then a single ndarray
            is generated, otherwise a tuple of two ndarrays. Note, that the
            batch arrays will contain the input elements in random order.
    """
    ndata = x.shape[0]
    indices = range(0, ndata)
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(indices)
    xr = x[indices]
    yr = y[indices] if y is not None else None
    for items in batch_generator(size, xr, yr, fixed_size):
        yield items
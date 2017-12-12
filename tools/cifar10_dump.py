#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import sys
if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib as urllib2
    import urlparse
import tarfile

def get_cifar10_data(folder, data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"):
    scheme, netloc, path, query, fragment = urlparse.urlsplit(data_url)
    filename = os.path.basename(path)
    if folder is not None:
        filename = os.path.join(folder, filename)
    if not os.path.isfile(filename):
        urllib2.urlretrieve(data_url, filename)
        tar = tarfile.open(filename, 'r')
        tar.extractall(path=folder)
        tar.close()
    cifar_folder = os.path.join(folder, "cifar-10-batches-py")
    return cifar_folder


def one_hot(arr, size, dtype=np.float32):
    arr_out = np.zeros([len(arr), size], dtype=dtype)
    for i in range(len(arr)):
        arr_out[i, arr[i]] = 1
    return arr_out


def convert(imgs, img_size=32, n_chans=3, data_format="channels_first"):
    imgs = imgs.astype(float) / 255.0
    imgs = imgs.reshape([-1, n_chans, img_size, img_size]) # NCHW
    if data_format == "channels_last":
        imgs = imgs.transpose([0, 2, 3, 1])    
    return imgs


def unpickle(f):
    with open(f, 'rb') as fo:
        if sys.version_info >= (3,):        
            d = pickle.load(fo, encoding="bytes")
        else:
            d = pickle.load(fo)
    return d


def main(out_folder = "/home/autasi/Work/gitTF/cifar10/data/", 
         data_format = "channels_first"):    
    cifar_folder = get_cifar10_data(out_folder)
    n_chans = 3
    img_size = 32
    tr_x = []
    tr_y = []
    for i in range(5):
        path = os.path.join(cifar_folder, "data_batch_" + str(i+1))
        d = unpickle(path)
        data = convert(d[b'data'], img_size=img_size, n_chans=n_chans, data_format=data_format)
        labels = one_hot(np.array(d[b'labels']), size=10)
        tr_x.append(data)
        tr_y.append(labels)
    tr_x = np.concatenate(tr_x)
    tr_y = np.concatenate(tr_y)
    
    path = os.path.join(cifar_folder, "test_batch")
    d = unpickle(path)
    te_x = convert(d[b'data'], img_size=img_size, n_chans=n_chans, data_format=data_format)
    te_y = one_hot(np.array(d[b'labels']), size=10)
    
    data = {'train': (tr_x, tr_y), 'test': (te_x, te_y)}
    if data_format == "channels_first":
        suf = "nchw"
    else:
        suf = "nhwc"
    fname = os.path.join(out_folder, "data_" + suf + ".pkl")
    with open(fname, "wb") as f:
        pickle.dump(data, f)
    

if __name__ == "__main__":
    main(data_format="channels_last")
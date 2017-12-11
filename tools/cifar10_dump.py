#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import cv2
import urllib.parse
import urllib.request
import tarfile

def get_cifar10_data(folder, data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"):
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(data_url)
    filename = os.path.basename(path)
    if folder is not None:
        filename = os.path.join(folder, filename)
    urllib.request.urlretrieve(data_url, filename)
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


def convert(imgs, img_size, data_format="channels_first"):
    imgs = imgs.reshape([len(imgs),img_size,img_size,3])
    if data_format == "channels_first":
        data = np.zeros([len(imgs), 1, img_size, img_size], dtype=np.float32)
    else:
        data = np.zeros([len(imgs), img_size, img_size, 1], dtype=np.float32)
    for i in range(len(imgs)):
        img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        if data_format == "channels_first":
            data[i,0,:,:] = (img/255.0).astype(data.dtype)
        else:
            data[i,:,:,0] = (img/255.0).astype(data.dtype)
    return data


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


def main(out_folder = "/home/ucu/Work/git/cifar10/data/", 
         data_format = "channels_first"):    
    cifar_folder = get_cifar10_data(out_folder)
    img_size = 32
    tr_x = []
    tr_y = []
    for i in range(5):
        path = os.path.join(cifar_folder, "data_batch_" + str(i+1))
        d = unpickle(path)
        data = convert(d[b'data'], img_size=img_size, data_format=data_format)
        labels = one_hot(np.array(d[b'labels']), size=10)
        tr_x.append(data)
        tr_y.append(labels)
    tr_x = np.concatenate(tr_x)
    tr_y = np.concatenate(tr_y)
    
    path = os.path.join(cifar_folder, "test_batch")
    d = unpickle(path)
    te_x = convert(d[b'data'], img_size=img_size, data_format=data_format)
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
# -*- coding: utf-8 -*-


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import os
from config import mnist_data_folder

def convert(imgs, img_size, data_format="channels_first"):
    if data_format == "channels_first":
        data = np.reshape(imgs, [imgs.shape[0], 1, img_size, img_size])
    else:
        data = np.reshape(imgs, [imgs.shape[0], img_size, img_size, 1])
    return data


def main(out_folder = "/home/ucu/Work/git/mnist/data/",
         data_format = "channels_first"):
    mnist = input_data.read_data_sets(out_folder, one_hot=True)
    img_size = 28
    tr_x = convert(mnist.train.images, img_size, data_format=data_format)
    tr_y = mnist.train.labels
    te_x = convert(mnist.test.images, img_size, data_format=data_format)
    te_y = mnist.test.labels
    val_x = convert(mnist.validation.images, img_size, data_format=data_format)
    val_y = mnist.validation.labels
    data = {'train': (tr_x, tr_y), 'test': (te_x, te_y), 'validation': (val_x, val_y)}
    if data_format == "channels_first":
        suf = "nchw"
    else:
        suf = "nhwc"
    fname = os.path.join(out_folder, "data_" + suf + ".pkl")
    with open(fname, "wb") as f:
        pickle.dump(data, f)
    

if __name__ == "__main__":
    main(out_folder=mnist_data_folder, data_format="channels_last")
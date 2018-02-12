#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arch.sequential_graph import cifar10_sequential_allconvC_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import pixel_mean_std
from arch.misc import DivideAt
import tensorflow as tf


#https://arxiv.org/pdf/1412.6806.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = pixel_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_sequential_allconvC_wd,
            optimizer = tf.train.MomentumOptimizer,
            optimizer_args = {'momentum': 0.9},
            n_epochs = 350,
            batch_size = 64,
            lr_decay_func = DivideAt(
                            start = 0.05,
                            divide_by = 10,
                            at_steps = [200, 250, 300]),
            weight_decay = 0.001
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  349
#Learning rate:  5e-05
#Test accuracy:  0.8893555
#Train accuracy:  0.933239
#Mean accuracy:  0.8893555


if __name__ == "__main__":
    main()

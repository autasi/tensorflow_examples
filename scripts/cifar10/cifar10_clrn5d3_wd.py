#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.sequential_graph import cifar10_sequential_clrn5d3_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import global_mean_std
from arch.misc import FixValue
import tensorflow as tf
from functools import partial


def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = partial(cifar10_sequential_clrn5d3_wd, drop_rate = 0.3),
            optimizer = tf.train.AdamOptimizer,
            optimizer_args = None,
            n_epochs = 100,
            batch_size = 128,
            lr_decay_func = FixValue(value = 1e-4),
            weight_decay = 0.0001
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  99
#Learning rate:  0.0001
#Test accuracy:  0.8318359
#Train accuracy:  0.94434386
#Mean accuracy:  0.8318359

if __name__ == "__main__":
    main()

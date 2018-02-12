#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arch.sequential_graph import cifar10_sequential_cbn6d_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import global_mean_std
from arch.misc import DivideAtRatesWithDecay
import tensorflow as tf


#https://www.kaggle.com/c/cifar-10/discussion/40237
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_sequential_cbn6d_wd,
            optimizer = tf.train.RMSPropOptimizer,
            optimizer_args = {'decay': 0.9, 'epsilon': 1e-8},
            n_epochs = 125,
            batch_size = 64,
            lr_decay_func = DivideAtRatesWithDecay(
                            start = 0.001,
                            divide_by = 2,
                            at = [0.6, 0.8],
                            max_steps = 125,
                            decay = 1e-6),
            weight_decay = 0.0001
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  124
#Learning rate:  0.00024996900384352343
#Test accuracy:  0.88964844
#Train accuracy:  0.9499801
#Mean accuracy:  0.88964844

if __name__ == "__main__":
    main()

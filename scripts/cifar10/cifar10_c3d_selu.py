#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arch.sequential_graph import cifar10_sequential_c3d_selu
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import global_mean_std
from arch.misc import ExponentialDecay
import tensorflow as tf

def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_sequential_c3d_selu,
            n_epochs = 50,
            batch_size = 128,
            lr_decay_func = ExponentialDecay(
                    start = 0.001,
                    stop = 0.0001,
                    max_steps = 50),
            optimizer = tf.train.AdamOptimizer,
            weight_decay = None,
            augmentation = False)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.00010471285480508996
#Test accuracy:  0.7595703
#Train accuracy:  1.0    



if __name__ == "__main__":
    main()
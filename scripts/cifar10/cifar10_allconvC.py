#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arch.sequential_graph import cifar10_sequential_allconvC
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import global_mean_std
from arch.misc import ExponentialDecay
import tensorflow as tf

def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_sequential_allconvC,
            n_epochs = 100,
            batch_size = 128,
            lr_decay_func = ExponentialDecay(
                    start = 0.001,
                    stop = 0.0001,
                    max_steps = 100),
            optimizer = tf.train.AdamOptimizer,
            weight_decay = None,
            augmentation = False)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  99
#Learning rate:  0.00010232929922807544
#Test accuracy:  0.8022461
#Train accuracy:  0.9791733
#Mean accuracy:  0.8022461    

if __name__ == "__main__":
    main()
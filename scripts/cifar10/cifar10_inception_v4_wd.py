#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arch.inception_graph import cifar10_inception_v4_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import channel_mean_std
from arch.misc import ExponentialDecayValue
import tensorflow as tf


# https://arxiv.org/pdf/1602.07261.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = channel_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_inception_v4_wd,
            optimizer = tf.train.RMSPropOptimizer,
            optimizer_args = {'momentum': 0.9, 'decay': 0.9, 'epsilon': 1.0},
            n_epochs = 100,
            batch_size = 32,
            lr_decay_func = ExponentialDecayValue(
                    start = 0.045,
                    decay_rate = 0.94,
                    decay_steps = 2),
            weight_decay = 0.00004)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  99
#Learning rate:  0.002103978351738769
#Test accuracy:  0.9405273
#Train accuracy:  1.0
#Mean accuracy:  0.9405273    

if __name__ == "__main__":
    main()


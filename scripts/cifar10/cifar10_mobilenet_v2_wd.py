#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.mobilenet_v2_graph import cifar10_mobilenet_v2_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import pixel_mean_std
from arch.misc import DecayValue
import tensorflow as tf


#https://arxiv.org/abs/1801.04381v2
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = pixel_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_mobilenet_v2_wd,
            optimizer = tf.train.RMSPropOptimizer,
            optimizer_args = {'decay': 0.9, 'momentum': 0.9},
            n_epochs = 300,
            batch_size = 96,
            lr_decay_func = DecayValue(
                            start = 0.045,
                            decay_rate = 0.98),
            weight_decay = 0.00004
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)

if __name__ == "__main__":
    main()

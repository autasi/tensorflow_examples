#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.sequential_graph import cifar10_sequential_c5d3_selu_drop_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import global_mean_std
from arch.misc import FixValue
import tensorflow as tf
from functools import partial


#https://arxiv.org/pdf/1706.02515.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = partial(cifar10_sequential_c5d3_selu_drop_wd, drop_rate = 0.15),
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
#Test accuracy:  0.8418945
#Train accuracy:  0.98711735
#Mean accuracy:  0.8418945  

if __name__ == "__main__":
    main()

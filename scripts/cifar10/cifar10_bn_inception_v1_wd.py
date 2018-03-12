#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arch.inception_graph import cifar10_bn_inception_v1_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import channel_mean_std
from arch.misc import ExponentialDecay
import tensorflow as tf

# http://proceedings.mlr.press/v37/ioffe15.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = channel_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_bn_inception_v1_wd,
            optimizer = tf.train.MomentumOptimizer,
            optimizer_args = {'momentum': 0.9},
            n_epochs = 250,
            batch_size = 128,
            lr_decay_func = ExponentialDecay(
                    start = 0.01,
                    stop = 0.0001,
                    max_steps = 200),
            weight_decay = 0.00004)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  249
#Learning rate:  0.0001
#Test accuracy:  0.89101565
#Train accuracy:  0.9992028
#Mean accuracy:  0.89101565    
    
    

if __name__ == "__main__":
    main()

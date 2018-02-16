#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.shufflenet_graph import cifar10_shufflenet_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import pixel_mean_std
from arch.misc import LinearDecay
import tensorflow as tf

#https://arxiv.org/pdf/1707.01083.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = pixel_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_shufflenet_wd,
            optimizer = tf.train.MomentumOptimizer,
            optimizer_args = {'momentum': 0.9},
            n_epochs = 300,
            batch_size = 128,
            lr_decay_func = LinearDecay(
                    start = 0.5,
                    stop = 0.0,
                    max_steps = 300),
            weight_decay = 4e-5
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  299
#Learning rate:  0.0016666666666666496
#Test accuracy:  0.8702148
#Train accuracy:  0.9992626
#Mean accuracy:  0.8702148

if __name__ == "__main__":
    main()

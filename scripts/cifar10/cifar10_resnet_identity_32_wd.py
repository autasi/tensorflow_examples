#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.resnet_identity_graph import cifar10_resnet_identity_32_wd
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import pixel_mean_std
from arch.misc import DivideAtRates
import tensorflow as tf

#https://arxiv.org/abs/1603.05027
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = pixel_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_resnet_identity_32_wd,
            optimizer = tf.train.MomentumOptimizer,
            optimizer_args = {'momentum': 0.9},
            n_epochs = 200,
            batch_size = 128,
            lr_decay_func = DivideAtRates(
                            start = 0.1,
                            divide_by = 10,
                            at = [0.5, 0.75],
                            max_steps = 200),
            weight_decay = 0.0001
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  199
#Learning rate:  0.001
#Test accuracy:  0.9277344
#Train accuracy:  0.99784756
#Mean accuracy:  0.9277344


if __name__ == "__main__":
    main()

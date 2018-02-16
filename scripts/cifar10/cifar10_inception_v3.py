#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.inception_graph import cifar10_inception_v3
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import global_mean_std
from arch.misc import ExponentialDecay
import tensorflow as tf

#https://arxiv.org/abs/1603.05027
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_inception_v3,
            optimizer = tf.train.AdamOptimizer,
            optimizer_args = None,
            n_epochs = 50,
            batch_size = 128,
            aux_loss_weight = 0.3,
            label_smoothing = 0.1,
            lr_decay_func = ExponentialDecay(start=0.01, stop=0.001, max_steps=50),
            weight_decay = None
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.0010471285480508996
#Test accuracy:  0.92441404
#Train accuracy:  0.9998007
#Mean accuracy:  0.92441404

if __name__ == "__main__":
    main()

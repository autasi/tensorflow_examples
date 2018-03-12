#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from arch.inception_graph import cifar10_inception_v1
from util.eval import load_cifar10_data, eval_net_custom
from util.normalization import global_mean_std
from arch.misc import ExponentialDecay
import tensorflow as tf

def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_custom(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_inception_v1,
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
#Test accuracy:  0.8734375
#Train accuracy:  1.0
#Mean accuracy:  0.8734375   

if __name__ == "__main__":
    main()


#from arch.inception_graph import cifar10_inception_v1
#from util.eval import load_cifar10_data, eval_net_basic
#from util.normalization import global_mean_std
#
#
##https://arxiv.org/pdf/1409.4842v1.pdf
#def main():
#    tr_x, tr_y, te_x, te_y = load_cifar10_data()
#    tr_x, te_x = global_mean_std(tr_x, te_x)
#    mean_acc, max_acc, min_acc = eval_net_basic(
#            tr_x, tr_y, te_x, te_y,
#            net_func = cifar10_inception_v1)
#    print("Mean accuracy: ", mean_acc)
#    print("Max accuracy: ", max_acc)
#    print("Min accuracy: ", min_acc)
##Epoch:  49
##Learning rate:  0.0010471285480508996
##Test accuracy:  0.7675781
##Train accuracy:  0.9723693
##Mean accuracy:  0.7675781    
#
#
#if __name__ == "__main__":
#    main()
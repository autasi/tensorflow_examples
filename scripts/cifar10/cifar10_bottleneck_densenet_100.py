#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arch.densenet_graph import cifar10_bottleneck_densenet_100
from util.eval import load_cifar10_data, eval_net_basic
from util.normalization import global_mean_std


# https://arxiv.org/pdf/1611.05431.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_basic(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_bottleneck_densenet_100
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.0010471285480508996
#Test accuracy:  0.8791016
#Train accuracy:  0.9813776
#Mean accuracy:  0.8791016    


if __name__ == "__main__":    
    main()



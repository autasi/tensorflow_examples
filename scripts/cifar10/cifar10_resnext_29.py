#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.resnext_graph import cifar10_resnext_29
from util.eval import load_cifar10_data, eval_net_basic
from util.normalization import global_mean_std
from functools import partial

#https://arxiv.org/pdf/1611.05431.pdf 
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_basic(
            tr_x, tr_y, te_x, te_y,
            net_func = partial(cifar10_resnext_29, cardinality = 8, group_width = 16)
            )
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.0010471285480508996
#Test accuracy:  0.87177736
#Train accuracy:  0.99988043
#Mean accuracy:  0.87177736


if __name__ == "__main__":    
    main()
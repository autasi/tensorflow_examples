#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.xception_graph import cifar10_xception
from util.eval import load_cifar10_data, eval_net_basic
from util.normalization import global_mean_std


# https://arxiv.org/pdf/1512.00567.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_basic(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_xception)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.0010471285480508996
#Test accuracy:  0.85117185
#Train accuracy:  0.99986047
#Mean accuracy:  0.85117185   

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.resnet_graph import cifar10_resnet_20
from util.eval import load_cifar10_data, eval_net_basic
from util.normalization import global_mean_std

def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_basic(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_resnet_20)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.0010471285480508996
#Test accuracy:  0.8208008
#Train accuracy:  0.99527663
#Mean accuracy:  0.8208008
    

if __name__ == "__main__":
    main()

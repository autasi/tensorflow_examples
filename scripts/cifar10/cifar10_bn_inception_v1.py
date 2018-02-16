#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.inception_graph import cifar10_bn_inception_v1
from util.eval import load_cifar10_data, eval_net_basic
from util.normalization import global_mean_std


# http://proceedings.mlr.press/v37/ioffe15.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_basic(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_bn_inception_v1)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.0010471285480508996
#Test accuracy:  0.90048826
#Train accuracy:  1.0
#Mean accuracy:  0.90048826    


if __name__ == "__main__":
    main()



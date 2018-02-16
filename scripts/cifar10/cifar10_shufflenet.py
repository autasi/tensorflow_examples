#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from arch.shufflenet_graph import cifar10_shufflenet
from util.eval import load_cifar10_data, eval_net_basic
from util.normalization import global_mean_std


#https://arxiv.org/pdf/1707.01083.pdf
def main():
    tr_x, tr_y, te_x, te_y = load_cifar10_data()
    tr_x, te_x = global_mean_std(tr_x, te_x)
    mean_acc, max_acc, min_acc = eval_net_basic(
            tr_x, tr_y, te_x, te_y,
            net_func = cifar10_shufflenet)
    print("Mean accuracy: ", mean_acc)
    print("Max accuracy: ", max_acc)
    print("Min accuracy: ", min_acc)
#Epoch:  49
#Learning rate:  0.0010471285480508996
#Test accuracy:  0.8748047
#Train accuracy:  0.99924266
#Mean accuracy:  0.8748047    
   

if __name__ == "__main__":
    main()


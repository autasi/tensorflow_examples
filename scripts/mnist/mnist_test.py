#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import tensorflow as tf
from arch.graph import mnist_sequential
from arch.io import load_variables
from util.misc import tuple_list_find
from util.batch import batch_generator


def main():
    # input data is in NHWC format
    data = pickle.load(open("/home/ucu/Work/git/mnist/data/data_nhwc.pkl", "rb"))
    te = data['test']
    te_x = te[0]
    te_y = te[1]

    height = te_x.shape[1]
    width = te_x.shape[2]
    n_chans = te_x.shape[3]
    n_classes = te_y.shape[1]
    
    # initialization
    tf.reset_default_graph()
    np.random.seed(42)
    tf.set_random_seed(42)    
    
    # input variables, image data + ground truth labels
    x = tf.placeholder(tf.float32, [None, height, width, n_chans], name="input")
    gt = tf.placeholder(tf.float32, [None, n_classes], name="label")
    
    # create network
    layers, variables = mnist_sequential(x)
    
    # training variable to control dropout
    training = tuple_list_find(variables, "training")[1]
    
    # logit output required for optimization
    logit = tuple_list_find(layers, "fc2")[1]
                
    # correct classifications
    corr = tf.equal(tf.argmax(logit, 1), tf.argmax(gt, 1))
    
    # accuray = average of correct classifications
    accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
    
    session = tf.Session()
    with session.as_default():
        # initialization of variables
        session.run(tf.global_variables_initializer())        
        load_variables(session, "/home/ucu/Work/git/mnist/network/mnist_expdecay.pkl")
        acc = []
        # evaluations on test set
        for (xb, yb) in batch_generator(512, te_x, te_y, fixed_size=False):
            ac = session.run(accuracy, feed_dict={x: xb,
                                                  gt: yb,
                                                  training: False})
            acc.append(ac)
        print("Test accuracy: ", np.mean(acc))    
    session.close()
    session = None
#Test accuracy:  0.984237    


if __name__ == "__main__":
    # environment variables for intel MKL
    os.environ["KMP_BLOCKTIME"] = str(0)
    os.environ["KMP_SETTINGS"] = str(1)
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    os.environ["OMP_NUM_THREADS"]= str(4)
    
    main()
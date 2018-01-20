#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import tensorflow as tf
from arch.resnet_graph import cifar10_resnet_20
from arch.misc import ExponentialDecay
from arch.io import save_variables
from util.misc import tuple_list_find
from util.batch import random_batch_generator, batch_generator
from config import cifar10_data_folder, cifar10_net_folder


def main():
    # input data is in NHWC format
    data_path = os.path.join(cifar10_data_folder, "data_nhwc.pkl")
    data = pickle.load(open(data_path, "rb"))
    tr = data['train']
    tr_x = tr[0]
    tr_y = tr[1]
    te = data['test']
    te_x = te[0]
    te_y = te[1]

    height = tr_x.shape[1]
    width = tr_x.shape[2]
    n_chans = tr_x.shape[3]
    n_classes = tr_y.shape[1]
    
    # data normalization
    eps = 1e-7
    tr_mean = np.mean(tr_x, axis = (0,1,2,3))
    tr_std = np.std(tr_x, axis = (0,1,2,3))
    tr_x = (tr_x-tr_mean)/(tr_std+eps)
    te_x = (te_x-tr_mean)/(tr_std+eps)    
    
    # initialization
    tf.reset_default_graph()
    np.random.seed(42)
    tf.set_random_seed(42)    
    
    # input variables, image data + ground truth labels
    x = tf.placeholder(tf.float32, [None, height, width, n_chans], name="input")
    gt = tf.placeholder(tf.float32, [None, n_classes], name="label")
    
    # create network
    layers, variables = cifar10_resnet_20(x)
    
    # training variable to control dropout
    training = tuple_list_find(variables, "training")[1]
    
    # logit output required for optimization
    logit = tuple_list_find(layers, "logit")[1]
        
    n_epochs = 50
    
    # optimization is done one the cross-entropy between labels and predicted logit    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=logit))
    
    # keeps track of the number of batches used for updating the network
    global_step = tf.Variable(0, trainable=False, name="global_step")
    
    # input variable for passing learning rates for the optimizer
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # some layers (e.g. batch normalization) require updates to internal variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy,
                                                                                  global_step = global_step)
        
    # correct classifications
    corr = tf.equal(tf.argmax(logit, 1), tf.argmax(gt, 1))
    
    # accuray = average of correct classifications
    accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
    
    # learning rate with exponential decay
    exp_decay = ExponentialDecay(start=0.01, stop=0.001, max_steps=50)

    session = tf.Session()
    with session.as_default():
        # initialization of variables
        session.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            lr = next(exp_decay)
            # training via random batches
            for (xb, yb) in random_batch_generator(128, tr_x, tr_y, seed = 42+i):
                session.run(train_step, feed_dict = {x: xb,
                                                     gt: yb,
                                                     training: True,
                                                     learning_rate: lr})

            tr_acc = []
            # evaluations on train set
            for (xb, yb) in batch_generator(256, tr_x, tr_y, fixed_size = False):
                ac = session.run(accuracy, feed_dict = {x: xb,
                                                        gt: yb,
                                                        training: False})
                tr_acc.append(ac)    
    
            acc = []
            # evaluations on test set
            for (xb, yb) in batch_generator(256, te_x, te_y, fixed_size = False):
                ac = session.run(accuracy, feed_dict = {x: xb,
                                                        gt: yb,
                                                        training: False})
                acc.append(ac)
            print("Epoch: ", i)
            print("Learning rate: ", lr)
            print("Test accuracy: ", np.mean(acc))
            print("Train accuracy: ", np.mean(tr_acc))            
        net_path = os.path.join(cifar10_net_folder, "cifar10_resnet20_expdecay.pkl")
        save_variables(session, net_path)
    session.close()
    session = None
#('Epoch: ', 49)
#('Learning rate: ', 0.0010471285480508996)
#('Test accuracy: ', 0.82851565)
#('Train accuracy: ', 0.99990034)


if __name__ == "__main__":
    # environment variables for intel MKL
    os.environ["KMP_BLOCKTIME"] = str(0)
    os.environ["KMP_SETTINGS"] = str(1)
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    os.environ["OMP_NUM_THREADS"]= str(4)
    
    main()
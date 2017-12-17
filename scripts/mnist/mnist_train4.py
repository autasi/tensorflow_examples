#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import tensorflow as tf
from functools import partial
from arch.graph import mnist_sequential_dbn2d1
from util.misc import tuple_list_find
from util.batch import random_batch_generator, batch_generator
from arch.io import save_variables
from config import mnist_data_folder, mnist_net_folder
from gp import bayesian_optimisation

# trains MNIST sequential network using bayesian optimization
def eval_network(params, data, save=False):
    learning_rate = 10.0**params[0]
    dropout_rate = params[1]

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

    # initialization
    tf.reset_default_graph()
    np.random.seed(42)
    tf.set_random_seed(42)    
    
    # input variables, image data + ground truth labels
    x = tf.placeholder(tf.float32, [None, height, width, n_chans], name="input")
    gt = tf.placeholder(tf.float32, [None, n_classes], name="label")
    
    # create network
    layers, variables = mnist_sequential_dbn2d1(x, drop_rate=dropout_rate)
    
    # training variable to control dropout
    training = tuple_list_find(variables, "training")[1]
    
    # logit output required for optimization
    logit = tuple_list_find(layers, "fc3")[1]
        
    n_epochs = 40
    
    # optimization is done one the cross-entropy between labels and predicted logit    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=logit))
    
    # keeps track of the number of batches used for updating the network
    global_step = tf.Variable(0, trainable=False, name="global_step")
    
    # some layers (e.g. batch normalization) require updates to internal variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy,
                                                                                  global_step = global_step)
        
    # correct classifications
    corr = tf.equal(tf.argmax(logit, 1), tf.argmax(gt, 1))
    
    # accuray = average of correct classifications
    accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
    
    session = tf.Session()
    mean_acc = None
    with session.as_default():
        # initialization of variables
        session.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            # training via random batches
            for (xb, yb) in random_batch_generator(256, tr_x, tr_y):
                session.run(train_step, feed_dict={x: xb,
                                                   gt: yb,
                                                   training: True})
            acc = []
            # evaluations on test set
            for (xb, yb) in batch_generator(512, te_x, te_y, fixed_size=False):
                ac = session.run(accuracy, feed_dict={x: xb,
                                                      gt: yb,
                                                      training: False})
                acc.append(ac)
            mean_acc = np.mean(acc)
            if save:
                print("Epoch: ", i)
                print("Learning rate: ", learning_rate)
                print("Drop rate: ", dropout_rate)
                print("Test accuracy: ", mean_acc)
        if save:
            net_path = os.path.join(mnist_net_folder, "mnist_dbn2d1_bayesopt.pkl")
            save_variables(session, net_path)            
    session.close()
    session = None
    return mean_acc


def main():
    data_path = os.path.join(mnist_data_folder, "data_nhwc.pkl")
    data = pickle.load(open(data_path, "rb"))
        
    bounds = np.array([[-5, -2], [0.1, 0.7]])
    np.random.seed(42)
    xp, yp = bayesian_optimisation(n_iters=30,
                                   sample_loss=partial(eval_network, data=data, save=False),
                                   bounds=bounds,
                                   n_pre_samples=5,
                                   random_search=100000)
    
    idx = yp.argmax()
    params = xp[idx]
    
    train = partial(eval_network, data=data, save=True)
    train(params)
#Epoch:  39
#Learning rate:  0.00458747955171
#Drop rate:  0.252330529702
#Test accuracy:  0.983295    
    

if __name__ == "__main__":
    # environment variables for intel MKL
    os.environ["KMP_BLOCKTIME"] = str(0)
    os.environ["KMP_SETTINGS"] = str(1)
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    os.environ["OMP_NUM_THREADS"]= str(4)    
    main()
    
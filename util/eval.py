#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf
import numpy as np
from config import cifar10_data_folder
from util.misc import tuple_list_find
from arch.misc import ExponentialDecay
from util.batch import random_batch_generator, batch_generator
from util.transform import RandomizedTransformer, Affine

def load_cifar10_data():
    data_path = os.path.join(cifar10_data_folder, "data_nhwc.pkl")
    data = pickle.load(open(data_path, "rb"))
    tr = data['train']
    tr_x = tr[0]
    tr_y = tr[1]
    te = data['test']
    te_x = te[0]
    te_y = te[1]
    return tr_x, tr_y, te_x, te_y


def _eval_net_custom(
        tr_x, tr_y, te_x, te_y,
        net_func,
        n_epochs,
        batch_size,
        lr_decay_func,
        optimizer,
        optimizer_args = None,
        weight_decay = 0.0,
        seed = 42):
    height = tr_x.shape[1]
    width = tr_x.shape[2]
    n_chans = tr_x.shape[3]
    n_classes = tr_y.shape[1]    

    if optimizer_args is None:
        optimizer_args = dict()

    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)    
    
    # input variables, image data + ground truth labels
    x = tf.placeholder(tf.float32, [None, height, width, n_chans], name="input")
    gt = tf.placeholder(tf.float32, [None, n_classes], name="label")
    
    # create network
    if weight_decay > 0.0:
        reg_weights = True
    else:
        reg_weights = False
    layers, variables = net_func(x, weight_decay = weight_decay, seed = seed)

    
    # training variable to control dropout
    training = tuple_list_find(variables, "training")[1]
    
    # logit output required for optimization
    logit = tuple_list_find(layers, "logit")[1]
    
    # optimization is done one the cross-entropy between labels and predicted logit    
    loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=logit))
    if reg_weights:
        reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_fn = loss_fn + tf.reduce_sum(reg_ws)
    
    
    # keeps track of the number of batches used for updating the network
    global_step = tf.Variable(0, trainable=False, name="global_step")
    
    # input variable for passing learning rates for the optimizer
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    optimizer_args.update({'learning_rate': learning_rate})
    
    # some layers (e.g. batch normalization) require updates to internal variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer(**optimizer_args).minimize(loss_fn, global_step = global_step)
        
    # correct classifications
    corr = tf.equal(tf.argmax(logit, 1), tf.argmax(gt, 1))
    
    # accuray = average of correct classifications
    accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
    
    # apply random affine transformations to training images
    transformer = RandomizedTransformer(
            transformer_class = Affine,
            params = [('shape', (height, width, n_chans)),
                      ('scale', 1.0)],
            rand_params = [('r', [-3.0, 3.0]),
                           ('tx', [-3.0, 3.0]),
                           ('ty', [-3.0, 3.0]),
                           ('reflect_y', [False, True])],
            mode = 'each',
            random_seed = seed)    
    
    acc_final = None
    session = tf.Session()
    with session.as_default():
        # initialization of variables
        session.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            lr = next(lr_decay_func)
            # training via random batches
            for (xb, yb) in random_batch_generator(batch_size, tr_x, tr_y, seed = seed+i):
                xbtr = np.zeros_like(xb)
                for j in range(len(xb)):
                    xbtr[j] = transformer.transform(xb[j])                
                session.run(train_step, feed_dict = {x: xbtr,
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
            acc_final = np.mean(acc)
    session.close()
    session = None
    return acc_final


def eval_net_custom(
        tr_x, tr_y, te_x, te_y,
        net_func,
        n_epochs,
        batch_size,
        lr_decay_func,
        optimizer,
        optimizer_args = None,
        weight_decay = 0.0,
        n_repeat = 1,
        seed = 42):

    accs = []
    for n in range(n_repeat):
        acc = _eval_net_custom(
                tr_x, tr_y, te_x, te_y,
                net_func,
                n_epochs,
                batch_size,
                lr_decay_func,
                optimizer,
                optimizer_args,
                weight_decay = weight_decay,
                seed = seed+n)
        accs.append(acc)
    return np.mean(accs), np.max(accs), np.min(accs)

def _eval_net_basic(
        tr_x, tr_y, te_x, te_y,
        net_func,
        n_epochs = 50,
        batch_size = 128,
        seed = 42):
    height = tr_x.shape[1]
    width = tr_x.shape[2]
    n_chans = tr_x.shape[3]
    n_classes = tr_y.shape[1]    


    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)    
    
    # input variables, image data + ground truth labels
    x = tf.placeholder(tf.float32, [None, height, width, n_chans], name="input")
    gt = tf.placeholder(tf.float32, [None, n_classes], name="label")
    
    # create network
    layers, variables = net_func(x, seed = seed)
    
    # training variable to control dropout
    training = tuple_list_find(variables, "training")[1]
    
    # logit output required for optimization
    logit = tuple_list_find(layers, "logit")[1]
    
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

    acc_final = None
    session = tf.Session()
    with session.as_default():
        # initialization of variables
        session.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            lr = next(exp_decay)
            # training via random batches
            for (xb, yb) in random_batch_generator(batch_size, tr_x, tr_y, seed = seed+i):
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
            acc_final = np.mean(acc)
    session.close()
    session = None
    return acc_final


def eval_net_basic(
        tr_x, tr_y, te_x, te_y,
        net_func,
        n_repeat = 1,
        n_epochs = 50,
        batch_size = 128,
        seed = 42):

    accs = []
    for n in range(n_repeat):
        acc = _eval_net_basic(
                tr_x, tr_y, te_x, te_y,
                net_func,
                n_epochs = 50,
                batch_size = batch_size,
                seed = seed+n)
        accs.append(acc)
    return np.mean(accs), np.max(accs), np.min(accs)
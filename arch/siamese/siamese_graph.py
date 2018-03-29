#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.layers import conv2d, conv2d_relu, conv2d_bn_relu, max_pool2d, dense_sigmoid, flatten, spatial_softmax
from arch.initializers import He_normal


def mnist_siamese_base(x, training = False, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    conv1 = conv2d_relu(
            x, size = 7, n_filters = 32,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name = "conv1")
    layers.append(("conv1", conv1))
    
    pool1 = max_pool2d(conv1, size = 2, stride = 2, name = "pool1")
    layers.append(("pool1", pool1)) # 14x14
    
    conv2 = conv2d_relu(
            pool1, size = 5, n_filters = 64,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2),
            name = "conv2")
    layers.append(("conv2", conv2))

    pool2 = max_pool2d(conv2, size = 2, stride = 2, name = "pool2")
    layers.append(("pool2", pool2)) # 7x7

    conv3 = conv2d_relu(
            pool2, size = 3, n_filters = 128,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+3),
            name = "conv3")
    layers.append(("conv3", conv3))

    pool3 = max_pool2d(conv3, size = 2, stride = 2, name = "pool3")
    layers.append(("pool3", pool3)) # 4x4

    conv4 = conv2d_relu(
            pool3, size = 1, n_filters = 256,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4),
            name = "conv4")
    layers.append(("conv4", conv4))

    pool4 = max_pool2d(conv4, size = 2, stride = 2, name = "pool4")
    layers.append(("pool4", pool4)) # 2x2

    conv5 = conv2d(
            pool4, size = 1, n_filters = 2,
            kernel_init = He_normal(seed = seed+5),
            name = "conv5")
    layers.append(("conv5", conv5))

    pool5 = max_pool2d(conv5, size = 2, stride = 2, name = "pool5")
    layers.append(("pool5", pool5)) # 1x1

    flat1 = flatten(pool5, name = "flatten1")
    layers.append(("output", flat1))
    
    return layers, variables

    
def mnist_siamese(x1, x2, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))
    
    
    with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE):
        # network 1
        layers1, variables1 = mnist_siamese_base(x1, training=training, weight_decay=weight_decay, seed=seed+0)
    
        # network 2
        layers2, variables2 = mnist_siamese_base(x2, training=training, weight_decay=weight_decay, seed=seed+len(layers1))
        
        layers.append(('output', (layers1, layers2)))
    
    return layers, variables    



def landmark_net(x, K = 8, training = False, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []
    
    conv1 = conv2d_bn_relu(
            x, size = 5, n_filters = 20,
            training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name = "conv1")
    layers.append(("conv1", conv1))

    pool1 = max_pool2d(conv1, size = 2, stride = 2, name = "pool1")
    layers.append(("pool1", pool1))

    conv2 = conv2d_bn_relu(
            pool1, size = 3, n_filters = 48,
            training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2),
            name = "conv2")
    layers.append(("conv2", conv2))

    conv3 = conv2d_bn_relu(
            conv2, size = 3, n_filters = 64,
            training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+3),
            name = "conv3")
    layers.append(("conv3", conv3))

    conv4 = conv2d_bn_relu(
            conv3, size = 3, n_filters = 80,
            training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4),
            name = "conv4")
    layers.append(("conv4", conv4))

    conv5 = conv2d_bn_relu(
            conv4, size = 3, n_filters = 256,
            training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+5),
            name = "conv5")
    layers.append(("conv5", conv5))

    conv6 = conv2d_bn_relu(
            conv5, size = 3, n_filters = K,
            training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+6),
            name = "conv6")
    layers.append(("conv6", conv6))
    
    sp_softmax = spatial_softmax(conv6, name = "spatial_softmax")
    #sp_softmax = tf.contrib.layers.spatial_softmax(conv6)
    layers.append(("output", sp_softmax))


    return layers, variables


def landmark_siamese(x1, x2, K = 8, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))
    
    
    with tf.variable_scope("siamese") as scope:    
        # network 1
        layers1, variables1 = landmark_net(x1, K=K, training=training, weight_decay=weight_decay, seed=seed+0)
    
        scope.reuse_variables()
        
        # network 2
        layers2, variables2 = mnist_siamese_base(x2, K=K, training=training, weight_decay=weight_decay, seed=seed+len(layers1))
        
        layers.append(('output', (layers1, layers2)))
    
    return layers, variables    



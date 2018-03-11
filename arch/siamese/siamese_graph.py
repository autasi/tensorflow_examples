#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.layers import conv2d_relu, max_pool2d, dense_sigmoid, flatten
from arch.initializers import He_normal


def mnist_siamese(x1, x2, n_blocks, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))
    
    
    with tf.variable_scope("siamese") as scope:    
        # network 1
        conv11 = conv2d_relu(
                x1, size = 10, n_filters = 64,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+1),
                name = "conv1")
        layers.append(("conv11", conv11))
        
        pool11 = max_pool2d(conv11, size = 2, stride = 2, name = "pool11")
        layers.append(("pool11", pool11))
        
        conv12 = conv2d_relu(
                pool11, size = 7, n_filters = 128,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+2),
                name = "conv2")
        layers.append(("conv12", conv12))
    
        pool12 = max_pool2d(conv12, size = 2, stride = 2, name = "pool12")
        layers.append(("pool12", pool12))
    
        conv13 = conv2d_relu(
                pool12, size = 4, n_filters = 128,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+3),
                name = "conv3")
        layers.append(("conv13", conv13))
    
        pool13 = max_pool2d(conv13, size = 2, stride = 2, name = "pool13")
        layers.append(("pool13", pool13))
    
        conv14 = conv2d_relu(
                pool13, size = 4, n_filters = 256,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+4),
                name = "conv4")
        layers.append(("conv14", conv14))
    
        flat1 = flatten(conv14, name = "flatten1")
        layers.append(("flatten1", flat1))
    
        dense11 = dense_sigmoid(
                flat1, n_units = 4096,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+5),
                name = "dense1")
        layers.append(("dense11", dense11))
    
        scope.reuse_variables()
    
        # network 2
        conv21 = conv2d_relu(
                x1, size = 10, n_filters = 64,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+6),
                name = "conv1")
        layers.append(("conv21", conv21))
    
        pool21 = max_pool2d(conv21, size = 2, stride = 2, name = "pool21")
        layers.append(("pool21", pool21))
        
        conv22 = conv2d_relu(
                pool21, size = 7, n_filters = 128,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+7),
                name = "conv2")
        layers.append(("conv22", conv22))
    
        pool22 = max_pool2d(conv22, size = 2, stride = 2, name = "pool22")
        layers.append(("pool22", pool22))
    
        conv23 = conv2d_relu(
                pool22, size = 4, n_filters = 128,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+8),
                name = "conv3")
        layers.append(("conv23", conv23))
    
        pool23 = max_pool2d(conv23, size = 2, stride = 2, name = "pool23")
        layers.append(("pool23", pool23))
    
        conv24 = conv2d_relu(
                pool23, size = 4, n_filters = 256,
                is_training = training,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+9),
                name = "conv4")
        layers.append(("conv24", conv24))
    
        flat2 = flatten(conv24, name = "flatten2")
        layers.append(("flatten2", flat2))
        
        dense21 = dense_sigmoid(
                flat2, n_units = 4096,
                regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                kernel_init = He_normal(seed = seed+10),
                name = "dense1")
        layers.append(("dense21", dense21))
    

    output = dense_sigmoid(
            merged, n_units = 1,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+11),
            name = "output")
    
    layers.append(("output", output))
            
    return layers, variables    
    
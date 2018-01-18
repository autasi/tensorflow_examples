#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from layers import conv2d_bn_relu, max_pool2d, global_avg_pool2d, dense
from initializers import He_normal, Kumar_normal
import inception


#https://arxiv.org/pdf/1409.4842v1.pdf
def cifar10_inception_v1(x, drop_rate = 0.4, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    # 32x32
    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))

    #3a - 32x32
    grid_3a = inception.grid(
                    conv,
                    n_filters_1x1 = 64,
                    n_reduce_3x3 = 96,
                    n_filters_3x3 = 128,
                    n_reduce_5x5 = 16,
                    n_filters_5x5 = 32,
                    n_filters_pool = 32,
                    kernel_init = He_normal(seed = seed+2),
                    kernel_init_reduce = He_normal(seed = seed+2),
                    name = "inception_grid_3a"
                    )
    layers.append(("grid_3a", grid_3a))
    
    #3b - 32x32
    grid_3b = inception.grid(
                    grid_3a,
                    n_filters_1x1 = 128,
                    n_reduce_3x3 = 128,
                    n_filters_3x3 = 192,
                    n_reduce_5x5 = 32,
                    n_filters_5x5 = 96,
                    n_filters_pool = 64,
                    kernel_init = He_normal(seed = seed+3),
                    kernel_init_reduce = He_normal(seed = seed+3),
                    name = "inception_grid_3a"
                    )
    layers.append(("grid_3b", grid_3b))


    pool1 = max_pool2d(grid_3b, size = 3, stride = 2, name = "pool1")
    layers.append(("pool1", pool1))
    
    #4a - 16x16
    grid_4a = inception.grid(
                    pool1,
                    n_filters_1x1 = 192,
                    n_reduce_3x3 = 96,
                    n_filters_3x3 = 208,
                    n_reduce_5x5 = 16,
                    n_filters_5x5 = 48,
                    n_filters_pool = 64,
                    kernel_init = He_normal(seed = seed+4),
                    kernel_init_reduce = He_normal(seed = seed+4),
                    name = "inception_grid_4a"
                    )
    layers.append(("grid_4a", grid_4a))
    
    #4d - 16x16
    grid_4d = inception.grid(
                    grid_4a,
                    n_filters_1x1 = 112,
                    n_reduce_3x3 = 144,
                    n_filters_3x3 = 288,
                    n_reduce_5x5 = 32,
                    n_filters_5x5 = 64,
                    n_filters_pool = 64,
                    kernel_init = He_normal(seed = seed+5),
                    kernel_init_reduce = He_normal(seed = seed+5),
                    name = "inception_grid_4d"
                    )
    layers.append(("grid_4d", grid_4d))


    pool2 = max_pool2d(grid_4d, size = 3, stride = 2, name = "pool2")
    layers.append(("pool2", pool2))

    # 5a - 8x8
    grid_5a = inception.grid(
                    pool2,
                    n_filters_1x1 = 256,
                    n_reduce_3x3 = 160,
                    n_filters_3x3 = 320,
                    n_reduce_5x5 = 32,
                    n_filters_5x5 = 128,
                    n_filters_pool = 128,
                    kernel_init = He_normal(seed = seed+6),
                    kernel_init_reduce = He_normal(seed = seed+6),
                    name = "inception_grid_5a" 
                    )
    layers.append(("grid_5a", grid_5a))

    # 5a - 8x8
    grid_5b = inception.grid(
                    grid_5a,
                    n_filters_1x1 = 384,
                    n_reduce_3x3 = 192,
                    n_filters_3x3 = 384,
                    n_reduce_5x5 = 48,
                    n_filters_5x5 = 128,
                    n_filters_pool = 128,
                    kernel_init = He_normal(seed = seed+7),
                    kernel_init_reduce = He_normal(seed = seed+7),
                    name = "inception_grid_5b" 
                    )
    layers.append(("grid_5b", grid_5b))

    pool3 = global_avg_pool2d(grid_5b, name = "pool3")
    layers.append(("pool3", pool3))
             
    dropout1 = tf.layers.dropout(
            pool3, rate = drop_rate, training = training, 
            seed = seed+8, name = "dropout")
    layers.append(("dropout1", dropout1))

    dense1 = dense(dropout1, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+9),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


# batch normalized version
def cifar10_bn_inception_v1(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    # 32x32
    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))

    #3a - 32x32
    grid_3a = inception.grid_fact2d_bn(
                    conv,
                    n_filters_1x1 = 64,
                    n_reduce_3x3 = 96,
                    n_filters_3x3 = 128,
                    n_reduce_5x5 = 16,
                    n_filters_5x5 = 32,
                    n_filters_pool = 32,
                    kernel_init = He_normal(seed = seed+2),
                    kernel_init_reduce = He_normal(seed = seed+2),
                    name = "inception_grid_3a_bn_fact"
                    )
    layers.append(("grid_3a", grid_3a))
    
    #3b - 32x32
    grid_3b = inception.grid_fact2d_bn(
                    grid_3a,
                    n_filters_1x1 = 128,
                    n_reduce_3x3 = 128,
                    n_filters_3x3 = 192,
                    n_reduce_5x5 = 32,
                    n_filters_5x5 = 96,
                    n_filters_pool = 64,
                    kernel_init = He_normal(seed = seed+3),
                    kernel_init_reduce = He_normal(seed = seed+3),
                    name = "inception_grid_3a_bn_fact"
                    )
    layers.append(("grid_3b", grid_3b))


    pool1 = max_pool2d(grid_3b, size = 3, stride = 2, name = "pool1")
    layers.append(("pool1", pool1))
    
    #4a - 16x16
    grid_4a = inception.grid_fact2d_bn(
                    pool1,
                    n_filters_1x1 = 192,
                    n_reduce_3x3 = 96,
                    n_filters_3x3 = 208,
                    n_reduce_5x5 = 16,
                    n_filters_5x5 = 48,
                    n_filters_pool = 64,
                    kernel_init = He_normal(seed = seed+4),
                    kernel_init_reduce = He_normal(seed = seed+4),
                    name = "inception_grid_4a_bn_fact"
                    )
    layers.append(("grid_4a", grid_4a))
    
    #4d - 16x16
    grid_4d = inception.grid_fact2d_bn(
                    grid_4a,
                    n_filters_1x1 = 112,
                    n_reduce_3x3 = 144,
                    n_filters_3x3 = 288,
                    n_reduce_5x5 = 32,
                    n_filters_5x5 = 64,
                    n_filters_pool = 64,
                    kernel_init = He_normal(seed = seed+5),
                    kernel_init_reduce = He_normal(seed = seed+5),
                    name = "inception_grid_4d_bn_fact"
                    )
    layers.append(("grid_4d", grid_4d))


    pool2 = max_pool2d(grid_4d, size = 3, stride = 2, name = "pool2")
    layers.append(("pool2", pool2))

    # 5a - 8x8
    grid_5a = inception.grid_fact2d_bn(
                    pool2,
                    n_filters_1x1 = 256,
                    n_reduce_3x3 = 160,
                    n_filters_3x3 = 320,
                    n_reduce_5x5 = 32,
                    n_filters_5x5 = 128,
                    n_filters_pool = 128,
                    kernel_init = He_normal(seed = seed+6),
                    kernel_init_reduce = He_normal(seed = seed+6),
                    name = "inception_grid_5a_bn_fact" 
                    )
    layers.append(("grid_5a", grid_5a))

    # 5a - 8x8
    grid_5b = inception.grid_fact2d_bn(
                    grid_5a,
                    n_filters_1x1 = 384,
                    n_reduce_3x3 = 192,
                    n_filters_3x3 = 384,
                    n_reduce_5x5 = 48,
                    n_filters_5x5 = 128,
                    n_filters_pool = 128,
                    kernel_init = He_normal(seed = seed+7),
                    kernel_init_reduce = He_normal(seed = seed+7),
                    name = "inception_grid_5b_bn_fact" 
                    )
    layers.append(("grid_5b", grid_5b))

    pool3 = global_avg_pool2d(grid_5b, name = "pool3")
    layers.append(("pool3", pool3))

    dense1 = dense(pool3, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+8),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d_bn_relu, conv2d_relu, max_pool2d, avg_pool2d, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal
from arch import inception


#https://arxiv.org/pdf/1409.4842v1.pdf
def cifar10_inception_v1(x, drop_rate = 0.4, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    # 32x32
    conv = conv2d_relu(
            x, size = 3, n_filters = 64,
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
                    name = "inception_grid_3b"
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

    dense1 = dense(
                dropout1, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+9),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


# http://proceedings.mlr.press/v37/ioffe15.pdf
# batch normalized version, and using factorized convolution
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
                    is_training = training,
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
                    is_training = training,
                    kernel_init = He_normal(seed = seed+3),
                    kernel_init_reduce = He_normal(seed = seed+3),
                    name = "inception_grid_3b_bn_fact"
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
                    is_training = training,
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
                    is_training = training,
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
                    is_training = training,
                    kernel_init = He_normal(seed = seed+6),
                    kernel_init_reduce = He_normal(seed = seed+6),
                    name = "inception_grid_5a_bn_fact" 
                    )
    layers.append(("grid_5a", grid_5a))

    # 5b - 8x8
    grid_5b = inception.grid_fact2d_bn(
                    grid_5a,
                    n_filters_1x1 = 384,
                    n_reduce_3x3 = 192,
                    n_filters_3x3 = 384,
                    n_reduce_5x5 = 48,
                    n_filters_5x5 = 128,
                    n_filters_pool = 128,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+7),
                    kernel_init_reduce = He_normal(seed = seed+7),
                    name = "inception_grid_5b_bn_fact" 
                    )
    layers.append(("grid_5b", grid_5b))

    pool3 = global_avg_pool2d(grid_5b, name = "pool3")
    layers.append(("pool3", pool3))

    dense1 = dense(
                pool3, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+8),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


#https://arxiv.org/pdf/1512.00567.pdf
#https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py
def cifar10_inception_v2(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    # 32x32x64
    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))

    #32x32x256
    grid_2a = inception.grid_fact2d_bn(
                    conv,
                    n_filters_1x1 = 64,
                    n_reduce_3x3 = 64,
                    n_filters_3x3 = 96,
                    n_reduce_5x5 = 48,
                    n_filters_5x5 = 64,
                    n_filters_pool = 32,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+2),
                    kernel_init_reduce = He_normal(seed = seed+2),
                    name = "inception_grid_2a_bn_fact"
                    )
    layers.append(("grid_2a", grid_2a))
    
    #32x32x256
    grid_2b = inception.grid_fact2d_bn(
                    grid_2a,
                    n_filters_1x1 = 64,
                    n_reduce_3x3 = 64,
                    n_filters_3x3 = 96,
                    n_reduce_5x5 = 48,
                    n_filters_5x5 = 64,
                    n_filters_pool = 32,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+3),
                    kernel_init_reduce = He_normal(seed = seed+3),
                    name = "inception_grid_2b_bn_fact"
                    )
    layers.append(("grid_2b", grid_2b))

    #16x16x768
    reduction1 = inception.reduction_bn(
                    grid_2b,
                    n_reduce_3x3_1 = 192,
                    n_filters_3x3_1 = 384,
                    n_reduce_3x3_2 = 64,
                    n_filters_3x3_2_1 = 96,
                    n_filters_3x3_2_2 = 96,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+4),
                    kernel_init_reduce = He_normal(seed = seed+4),
                    name = "reduction_1")
    layers.append(("reduction1", reduction1))
    
    #16x16x768
    grid_4a = inception.grid_fact1d_bn(
                    reduction1,
                    n_filters_1x1 = 192,
                    n_reduce_7x7_1 = 128,
                    n_filters_7x7_1 = [(128, 192)],
                    n_reduce_7x7_2 = 128,
                    n_filters_7x7_2 = [(128, 128),(128, 192)],
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+5),
                    kernel_init_reduce = He_normal(seed = seed+5),
                    name = "inception_grid_4a_bn_fact"
                    )
    layers.append(("grid_4a", grid_4a))
    
    #16x16x768
    grid_4b = inception.grid_fact1d_bn(
                    grid_4a,
                    n_filters_1x1 = 192,
                    n_reduce_7x7_1 = 160,
                    n_filters_7x7_1 = [(160, 192)],
                    n_reduce_7x7_2 = 160,
                    n_filters_7x7_2 = [(160, 160),(160, 192)],
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+6),
                    kernel_init_reduce = He_normal(seed = seed+6),
                    name = "inception_grid_4b_bn_fact"
                    )
    layers.append(("grid_4b", grid_4b))


    #8x8x1280
    reduction2 = inception.reduction_bn(
                    grid_4b,
                    n_reduce_3x3_1 = 192,
                    n_filters_3x3_1 = 320,
                    n_reduce_3x3_2 = 192,
                    n_filters_3x3_2_1 = 192,
                    n_filters_3x3_2_2 = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+7),
                    kernel_init_reduce = He_normal(seed = seed+7),
                    name = "reduction_2")
    layers.append(("reduction2", reduction2))

    #8x8x2048
    fb_6a = inception.expanded_filterbank_bn(
                    reduction2,
                    n_filters_1x1 = 320,
                    n_reduce_1x3_3x1 = 384,
                    n_filters_1x3_3x1 = 384,
                    n_reduce_3x3 = 448,
                    n_filters_3x3 = 384,
                    n_filters_3x3_1x3_3x1 = 384,
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+8),
                    kernel_init_reduce = He_normal(seed = seed+8),
                    name = "inception_filterbank_6a_bn" 
                    )
    layers.append(("fb_6a", fb_6a))

    #8x8x2048
    fb_6b = inception.expanded_filterbank_bn(
                    fb_6a,
                    n_filters_1x1 = 320,
                    n_reduce_1x3_3x1 = 384,
                    n_filters_1x3_3x1 = 384,
                    n_reduce_3x3 = 448,
                    n_filters_3x3 = 384,
                    n_filters_3x3_1x3_3x1 = 384,
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+9),
                    kernel_init_reduce = He_normal(seed = seed+9),
                    name = "inception_filterbank_6b_bn" 
                    )
    layers.append(("fb_6b", fb_6b))

    pool1 = global_avg_pool2d(fb_6b, name = "pool1")
    layers.append(("pool1", pool1))

    dense1 = dense(
                pool1, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+10),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


#https://arxiv.org/pdf/1512.00567.pdf
#https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py
# same as v2 + bn_auxiliary + label smoothing
def cifar10_inception_v3(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    # 32x32x64
    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))

    #32x32x256
    grid_2a = inception.grid_fact2d_bn(
                    conv,
                    n_filters_1x1 = 64,
                    n_reduce_3x3 = 64,
                    n_filters_3x3 = 96,
                    n_reduce_5x5 = 48,
                    n_filters_5x5 = 64,
                    n_filters_pool = 32,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+2),
                    kernel_init_reduce = He_normal(seed = seed+2),
                    name = "inception_grid_2a_bn_fact"
                    )
    layers.append(("grid_2a", grid_2a))
    
    #32x32x256
    grid_2b = inception.grid_fact2d_bn(
                    grid_2a,
                    n_filters_1x1 = 64,
                    n_reduce_3x3 = 64,
                    n_filters_3x3 = 96,
                    n_reduce_5x5 = 48,
                    n_filters_5x5 = 64,
                    n_filters_pool = 32,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+3),
                    kernel_init_reduce = He_normal(seed = seed+3),
                    name = "inception_grid_2b_bn_fact"
                    )
    layers.append(("grid_2b", grid_2b))

    #16x16x768
    reduction1 = inception.reduction_bn(
                    grid_2b,
                    n_reduce_3x3_1 = 192,
                    n_filters_3x3_1 = 384,
                    n_reduce_3x3_2 = 64,
                    n_filters_3x3_2_1 = 96,
                    n_filters_3x3_2_2 = 96,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+4),
                    kernel_init_reduce = He_normal(seed = seed+4),
                    name = "reduction_1")
    layers.append(("reduction1", reduction1))
    
    #16x16x768
    grid_4a = inception.grid_fact1d_bn(
                    reduction1,
                    n_filters_1x1 = 192,
                    n_reduce_7x7_1 = 128,
                    n_filters_7x7_1 = [(128, 192)],
                    n_reduce_7x7_2 = 128,
                    n_filters_7x7_2 = [(128, 128),(128, 192)],
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+5),
                    kernel_init_reduce = He_normal(seed = seed+5),
                    name = "inception_grid_4a_bn_fact"
                    )
    layers.append(("grid_4a", grid_4a))
    
    #16x16x768
    grid_4b = inception.grid_fact1d_bn(
                    grid_4a,
                    n_filters_1x1 = 192,
                    n_reduce_7x7_1 = 160,
                    n_filters_7x7_1 = [(160, 192)],
                    n_reduce_7x7_2 = 160,
                    n_filters_7x7_2 = [(160, 160),(160, 192)],
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+6),
                    kernel_init_reduce = He_normal(seed = seed+6),
                    name = "inception_grid_4b_bn_fact"
                    )
    layers.append(("grid_4b", grid_4b))

    # bn-auxiliary classifier
    aux = inception.auxiliary_classifier_bn(
                    grid_4b,
                    is_training = training,
                    kernel_init_conv = He_normal(seed = seed+7),
                    kernel_init_dense = Kumar_normal(activation = "relu", mode = "FAN_AVG", seed = seed+7),
                    name = "inception_auxiliary_classifier")
    layers.append(("aux", aux))
    
    aux_dense = dense(
                    aux, n_units = 10,
                    kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+8),
                    name = "aux_dense")
    layers.append(("aux_logit", aux_dense))
    
    aux_prob = tf.nn.softmax(aux_dense, name = "aux_prob")
    layers.append(("aux_prob", aux_prob))     

    #8x8x1280
    reduction2 = inception.reduction_bn(
                    grid_4b,
                    n_reduce_3x3_1 = 192,
                    n_filters_3x3_1 = 320,
                    n_reduce_3x3_2 = 192,
                    n_filters_3x3_2_1 = 192,
                    n_filters_3x3_2_2 = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+7),
                    kernel_init_reduce = He_normal(seed = seed+7),
                    name = "reduction_2")
    layers.append(("reduction2", reduction2))

    #8x8x2048
    fb_6a = inception.expanded_filterbank_bn(
                    reduction2,
                    n_filters_1x1 = 320,
                    n_reduce_1x3_3x1 = 384,
                    n_filters_1x3_3x1 = 384,
                    n_reduce_3x3 = 448,
                    n_filters_3x3 = 384,
                    n_filters_3x3_1x3_3x1 = 384,
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+8),
                    kernel_init_reduce = He_normal(seed = seed+8),
                    name = "inception_filterbank_6a_bn" 
                    )
    layers.append(("fb_6a", fb_6a))

    #8x8x2048
    fb_6b = inception.expanded_filterbank_bn(
                    fb_6a,
                    n_filters_1x1 = 320,
                    n_reduce_1x3_3x1 = 384,
                    n_filters_1x3_3x1 = 384,
                    n_reduce_3x3 = 448,
                    n_filters_3x3 = 384,
                    n_filters_3x3_1x3_3x1 = 384,
                    n_filters_pool = 192,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+9),
                    kernel_init_reduce = He_normal(seed = seed+9),
                    name = "inception_filterbank_6b_bn" 
                    )
    layers.append(("fb_6b", fb_6b))

    pool1 = global_avg_pool2d(fb_6b, name = "pool1")
    layers.append(("pool1", pool1))

    dense1 = dense(
                pool1, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+10),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


# Szegedy et al., Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
# https://arxiv.org/pdf/1602.07261.pdf
def cifar10_inception_v4(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    # 32x32x64
    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))

    #32x32x384
    grid_2a = inception.grid_fact2d_bn(
                    conv,
                    n_filters_1x1 = 96,
                    n_reduce_3x3 = 64,
                    n_filters_3x3 = 96,
                    n_reduce_5x5 = 64,
                    n_filters_5x5 = 96,
                    n_filters_pool = 96,
                    pool = avg_pool2d,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+2),
                    kernel_init_reduce = He_normal(seed = seed+2),
                    name = "inception_grid_2a_bn_fact"
                    )
    layers.append(("grid_2a", grid_2a))
    
    #32x32x384
    grid_2b = inception.grid_fact2d_bn(
                    grid_2a,
                    n_filters_1x1 = 96,
                    n_reduce_3x3 = 64,
                    n_filters_3x3 = 96,
                    n_reduce_5x5 = 64,
                    n_filters_5x5 = 96,
                    n_filters_pool = 96,
                    pool = avg_pool2d,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+3),
                    kernel_init_reduce = He_normal(seed = seed+3),
                    name = "inception_grid_2b_bn_fact"
                    )
    layers.append(("grid_2b", grid_2b))

    #16x16x1024
    reduction1 = inception.reduction_bn_v4_1(
                    grid_2b,
                    n_filters_3x3_1 = 384,
                    n_reduce_3x3_2 = 192,
                    n_filters_3x3_2_1 = 224,
                    n_filters_3x3_2_2 = 256,
                    padding = "SAME",
                    is_training = training,
                    kernel_init = He_normal(seed = seed+4),
                    kernel_init_reduce = He_normal(seed = seed+4),
                    name = "reduction_1")
    layers.append(("reduction1", reduction1))
    
    #16x16x1024
    grid_4a = inception.grid_fact1d_bn(
                    reduction1,
                    n_filters_1x1 = 384,
                    n_reduce_7x7_1 = 192,
                    n_filters_7x7_1 = [(224, 256)],
                    n_reduce_7x7_2 = 192,
                    n_filters_7x7_2 = [(192, 224),(224, 256)],
                    n_filters_pool = 128,
                    pool = avg_pool2d,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+5),
                    kernel_init_reduce = He_normal(seed = seed+5),
                    name = "inception_grid_4a_bn_fact"
                    )
    layers.append(("grid_4a", grid_4a))
    
    #16x16x1024
    grid_4b = inception.grid_fact1d_bn(
                    grid_4a,
                    n_filters_1x1 = 384,
                    n_reduce_7x7_1 = 192,
                    n_filters_7x7_1 = [(224, 256)],
                    n_reduce_7x7_2 = 192,
                    n_filters_7x7_2 = [(192, 224),(224, 256)],
                    n_filters_pool = 128,
                    pool = avg_pool2d,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+6),
                    kernel_init_reduce = He_normal(seed = seed+6),
                    name = "inception_grid_4b_bn_fact"
                    )
    layers.append(("grid_4b", grid_4b))

    #8x8x1536
    reduction2 = inception.reduction_bn_v4_2(
                    grid_4b,
                    n_reduce_3x3 = 192,
                    n_filters_3x3 = 192,
                    n_reduce_7x7 = 256,
                    n_filters_1x7 = 256,
                    n_filters_7x1 = 320,
                    n_filters_7x7_3x3 = 320,
                    padding = "SAME",
                    is_training = training,
                    kernel_init = He_normal(seed = seed+7),
                    kernel_init_reduce = He_normal(seed = seed+7),
                    name = "reduction_2")
    layers.append(("reduction2", reduction2))

    #8x8x1536
    fb_6a = inception.expanded_filterbank_fact1d_bn(
                    reduction2,
                    n_filters_1x1 = 256,
                    n_reduce_1x3_3x1 = 384,
                    n_filters_1x3_3x1 = 256,
                    n_reduce_1x3 = 384,
                    n_filters_1x3 = 448,
                    n_filters_3x1 = 512,
                    n_filters_3x3_1x3_3x1 = 256,
                    n_filters_pool = 256,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+8),
                    kernel_init_reduce = He_normal(seed = seed+8),
                    name = "inception_filterbank_6a_bn" 
                    )
    layers.append(("fb_6a", fb_6a))

    #8x8x1536
    fb_6b = inception.expanded_filterbank_fact1d_bn(
                    fb_6a,
                    n_filters_1x1 = 256,
                    n_reduce_1x3_3x1 = 384,
                    n_filters_1x3_3x1 = 256,
                    n_reduce_1x3 = 384,
                    n_filters_1x3 = 448,
                    n_filters_3x1 = 512,
                    n_filters_3x3_1x3_3x1 = 256,
                    n_filters_pool = 256,
                    is_training = training,
                    kernel_init = He_normal(seed = seed+9),
                    kernel_init_reduce = He_normal(seed = seed+9),
                    name = "inception_filterbank_6b_bn" 
                    )
    layers.append(("fb_6b", fb_6b))

    pool1 = global_avg_pool2d(fb_6b, name = "pool1")
    layers.append(("pool1", pool1))

    dense1 = dense(
                pool1, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+10),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables









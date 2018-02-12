# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d_bn_relu, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal
from arch import shufflenet

#Zhang et al., ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
#https://arxiv.org/pdf/1707.01083.pdf
def cifar10_shufflenet(x, n_groups = 2, n_filters = 200, ratio = 1.0, seed = 42):    
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 24,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
    
    slayer1 = shufflenet.shufflenet_layer(
            conv, n_filters = n_filters,
            n_repeat = 3, n_groups = n_groups,
            reduction_ratio = ratio,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "shufflenet_layer_1")
    layers.append(("shufflenet_layer_1", slayer1))
            
    slayer2 = shufflenet.shufflenet_layer(
            slayer1, n_filters = n_filters*2,
            n_repeat = 7, n_groups = n_groups,
            reduction_ratio = ratio,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "shufflenet_layer_2")
    layers.append(("shufflenet_layer_2", slayer2))

    slayer3 = shufflenet.shufflenet_layer(
            slayer2, n_filters = n_filters*4,
            n_repeat = 3, n_groups = n_groups,
            reduction_ratio = ratio,
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "shufflenet_layer_3")
    layers.append(("shufflenet_layer_3", slayer3))

    pool = global_avg_pool2d(slayer3)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


def cifar10_shufflenet_wd(x, n_groups = 2, n_filters = 200, ratio = 1.0,  weight_decay = 4e-5, seed = 42):    
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 24,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
    
    slayer1 = shufflenet.shufflenet_layer(
            conv, n_filters = n_filters,
            n_repeat = 3, n_groups = n_groups,
            reduction_ratio = ratio,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2),
            name = "shufflenet_layer_1")
    layers.append(("shufflenet_layer_1", slayer1))
            
    slayer2 = shufflenet.shufflenet_layer(
            slayer1, n_filters = n_filters*2,
            n_repeat = 7, n_groups = n_groups,
            reduction_ratio = ratio,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+3),
            name = "shufflenet_layer_2")
    layers.append(("shufflenet_layer_2", slayer2))

    slayer3 = shufflenet.shufflenet_layer(
            slayer2, n_filters = n_filters*4,
            n_repeat = 3, n_groups = n_groups,
            reduction_ratio = ratio,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4),
            name = "shufflenet_layer_3")
    layers.append(("shufflenet_layer_3", slayer3))

    pool = global_avg_pool2d(slayer3)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
#            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables
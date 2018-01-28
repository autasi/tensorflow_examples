#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from functools import partial
import tensorflow as tf
from arch.layers import conv2d_bn_act, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal
from arch import resnext

def cifar10_resnext_29(x, cardinality = 8, group_width = 64, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_bn_act(
            x, size = 3, n_filters = 64,
            activation = tf.nn.relu,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnext.residual_layer(
            conv, n_blocks = 3, stride = 1,
            cardinality = cardinality,
            group_width = group_width,
            block_function = resnext.bottleneck_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "residual_1")
    layers.append(("residual_1", res1))

    res2 = resnext.residual_layer(
            res1, n_blocks = 3, stride = 2,
            cardinality = cardinality,
            group_width = group_width*2,
            block_function = resnext.bottleneck_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),            
            name="residual_2")
    layers.append(("residual_2", res2))

    res3 = resnext.residual_layer(
            res2, n_blocks = 3, stride = 2,
            cardinality = cardinality,
            group_width = group_width*4,
            block_function = resnext.bottleneck_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "residual_3")
    layers.append(("residual_3", res3))

    pool = global_avg_pool2d(res3)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


def cifar10_resnext_29_wd(x, cardinality = 8, group_width = 64, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_bn_act(
            x, size = 3, n_filters = 64,
            activation = tf.nn.relu,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnext.residual_layer(
            conv, n_blocks = 3, stride = 1,
            cardinality = cardinality,
            group_width = group_width,
            block_function = partial(
                    resnext.bottleneck_block,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "residual_1")
    layers.append(("residual_1", res1))

    res2 = resnext.residual_layer(
            res1, n_blocks = 3, stride = 2,
            cardinality = cardinality,
            group_width = group_width*2, 
            block_function = partial(
                    resnext.bottleneck_block,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+3),            
            name="residual_2")
    layers.append(("residual_2", res2))

    res3 = resnext.residual_layer(
            res2, n_blocks = 3, stride = 2,
            cardinality = cardinality,
            group_width = group_width*4,
            block_function = partial(
                    resnext.bottleneck_block,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "residual_3")
    layers.append(("residual_3", res3))

    pool = global_avg_pool2d(res3)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables



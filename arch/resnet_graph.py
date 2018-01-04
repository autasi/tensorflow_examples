#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from functools import partial
import tensorflow as tf
from arch.layers import conv2d_act_bn, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal
from arch import resnet


def cifar10_resnet_20(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_act_bn(
            x, size = 3, n_filters = 16,
            activation = tf.nn.relu,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnet.residual_layer(
            conv, n_filters = 16, n_blocks = 3, stride = 1,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "residual_1")
    layers.append(("residual_1", res1))

    res2 = resnet.residual_layer(
            res1, n_filters = 32, n_blocks = 3, stride = 2,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),            
            name="residual_2")
    layers.append(("residual_2", res2))

    res3 = resnet.residual_layer(
            res2, n_filters = 64, n_blocks = 3, stride = 2,
            block_function = resnet.residual_block,
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


def cifar10_resnet_32(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_act_bn(
            x, size = 3, n_filters = 16,
            activation = tf.nn.relu,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnet.residual_layer(
            conv, n_filters = 16, n_blocks = 5, stride = 1,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "residual_1")
    layers.append(("residual_1", res1))

    res2 = resnet.residual_layer(
            res1, n_filters = 32, n_blocks = 5, stride = 2,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),            
            name="residual_2")
    layers.append(("residual_2", res2))

    res3 = resnet.residual_layer(
            res2, n_filters = 64, n_blocks = 5, stride = 2,
            block_function = resnet.residual_block,
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


def cifar10_resnet_bottleneck_20(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_act_bn(
            x, size = 3, n_filters = 16,
            activation = tf.nn.relu,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnet.residual_layer(
            conv, n_filters = 16, n_blocks = 3, stride = 1,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "residual_1")
    layers.append(("residual_1", res1))

    res2 = resnet.residual_layer(
            res1, n_filters = 32, n_blocks = 3, stride = 2,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),            
            name="residual_2")
    layers.append(("residual_2", res2))

    res3 = resnet.residual_layer(
            res2, n_filters = 64, n_blocks = 3, stride = 2,
            block_function = partial(resnet.bottleneck_block, n_filters_reduce = 16),
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


def cifar10_resnet_bottleneck_32(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_act_bn(
            x, size = 3, n_filters = 16,
            activation = tf.nn.relu,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnet.residual_layer(
            conv, n_filters = 16, n_blocks = 5, stride = 1,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "residual_1")
    layers.append(("residual_1", res1))

    res2 = resnet.residual_layer(
            res1, n_filters = 32, n_blocks = 5, stride = 2,
            block_function = resnet.residual_block,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),            
            name="residual_2")
    layers.append(("residual_2", res2))

    res3 = resnet.residual_layer(
            res2, n_filters = 64, n_blocks = 5, stride = 2,
            block_function = partial(resnet.bottleneck_block, n_filters_reduce = 16),
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


def cifar10_resnet_20_wd(x, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))
    
    conv = conv2d_act_bn(
            x, size = 3, n_filters = 16,
            activation = tf.nn.relu,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnet.residual_layer(
            conv, n_filters = 16, n_blocks = 3, stride = 1,
            block_function = partial(
                    resnet.residual_block,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)),
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "residual_1")
    layers.append(("residual_1", res1))

    res2 = resnet.residual_layer(
            res1, n_filters = 32, n_blocks = 3, stride = 2,
            block_function = partial(
                    resnet.residual_block,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)),
            is_training = training,
            kernel_init = He_normal(seed = seed+3),            
            name="residual_2")
    layers.append(("residual_2", res2))

    res3 = resnet.residual_layer(
            res2, n_filters = 64, n_blocks = 3, stride = 2,
            block_function = partial(
                    resnet.residual_block,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)),
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

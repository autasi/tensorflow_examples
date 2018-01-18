#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from functools import partial
import tensorflow as tf
from arch.layers import conv2d_act_bn, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal
from arch import resnet
from arch import resnext
from arch import senet

def cifar10_se_resnet_20(x, ratio = 8, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_act_bn(
            x, size = 3, n_filters = 16,
            activation = tf.nn.relu,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnet.residual_layer(
            conv, n_filters = 16, n_blocks = 3, stride = 1,
            block_function = partial(
                    senet.se_resnet_residual_block,
                    ratio = ratio,
                    se_kernel_init_1 = He_normal(seed = seed+2),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+2),
                    ), 
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "se_residual_1")
    layers.append(("se_residual_1", res1))

    res2 = resnet.residual_layer(
            res1, n_filters = 32, n_blocks = 3, stride = 2,
            block_function = partial(
                    senet.se_resnet_residual_block,
                    ratio = ratio,
                    se_kernel_init_1 = He_normal(seed = seed+3),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+3),
                    ), 
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "se_residual_2")
    layers.append(("se_residual_2", res2))

    res3 = resnet.residual_layer(
            res2, n_filters = 64, n_blocks = 3, stride = 2,
            block_function = partial(
                    senet.se_resnet_residual_block,
                    ratio = ratio,
                    se_kernel_init_1 = He_normal(seed = seed+4),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+4),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "se_residual_3")
    layers.append(("se_residual_3", res3))

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


def cifar10_se_resnet_20_wd(x, ratio = 8, weight_decay = 0.0001, seed = 42):
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
            name = "initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnet.residual_layer(
            conv, n_filters = 16, n_blocks = 3, stride = 1,
            block_function = partial(
                    senet.se_resnet_residual_block,
                    ratio = ratio,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    se_kernel_init_1 = He_normal(seed = seed+2),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+2),
                    ), 
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "se_residual_1")
    layers.append(("se_residual_1", res1))

    res2 = resnet.residual_layer(
            res1, n_filters = 32, n_blocks = 3, stride = 2,
            block_function = partial(
                    senet.se_resnet_residual_block,
                    ratio = ratio,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    se_kernel_init_1 = He_normal(seed = seed+3),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+3),
                    ), 
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "se_residual_2")
    layers.append(("se_residual_2", res2))

    res3 = resnet.residual_layer(
            res2, n_filters = 64, n_blocks = 3, stride = 2,
            block_function = partial(
                    senet.se_resnet_residual_block,
                    ratio = ratio,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    se_kernel_init_1 = He_normal(seed = seed+4),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+4),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "se_residual_3")
    layers.append(("se_residual_3", res3))

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


def cifar10_se_resnext_29(x, ratio = 16, seed = 42):

    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_act_bn(
            x, size = 3, n_filters = 64,
            activation = tf.nn.relu,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnext.residual_layer(
            conv, n_blocks = 3, stride = 1,
            n_filters = 64,
            cardinality = 8,
            group_width = 64,
            block_function = partial(
                    senet.senet.se_resnext_bottleneck_block,
                    ratio = ratio,
                    se_kernel_init_1 = He_normal(seed = seed+2),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+2),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "se_residual_1")
    layers.append(("se_residual_1", res1))

    res2 = resnext.residual_layer(
            res1, n_blocks = 3, stride = 2,
            n_filters = 128,
            cardinality = 8,
            group_width = 64,
            block_function = partial(
                    senet.senet.se_resnext_bottleneck_block,
                    ratio = ratio,
                    se_kernel_init_1 = He_normal(seed = seed+3),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+3),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "se_residual_2")
    layers.append(("se_residual_2", res2))

    res3 = resnext.residual_layer(
            res2, n_blocks = 3, stride = 2,
            n_filters = 256,
            cardinality = 8,
            group_width = 64,
            block_function = partial(
                    senet.senet.se_resnext_bottleneck_block,
                    ratio = ratio,
                    se_kernel_init_1 = He_normal(seed = seed+4),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+4),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "se_residual_3")
    layers.append(("se_residual_3", res3))

    pool = global_avg_pool2d(res3)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables


def cifar10_se_resnext_29_wd(x, ratio = 16, weight_decay = 0.0001, seed = 42):

    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_act_bn(
            x, size = 3, n_filters = 64,
            activation = tf.nn.relu,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
            
    res1 = resnext.residual_layer(
            conv, n_blocks = 3, stride = 1,
            n_filters = 64,
            cardinality = 16,
            group_width = 64,
            block_function = partial(
                    senet.se_resnext_bottleneck_block,
                    ratio = ratio,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    se_kernel_init_1 = He_normal(seed = seed+2),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+2),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "se_residual_1")
    layers.append(("se_residual_1", res1))

    res2 = resnext.residual_layer(
            res1, n_blocks = 3, stride = 2,
            n_filters = 128,
            cardinality = 16,
            group_width = 64,
            block_function = partial(
                    senet.se_resnext_bottleneck_block,
                    ratio = ratio,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    se_kernel_init_1 = He_normal(seed = seed+3),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+3),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "se_residual_2")
    layers.append(("se_residual_2", res2))

    res3 = resnext.residual_layer(
            res2, n_blocks = 3, stride = 2,
            n_filters = 256,
            cardinality = 16,
            group_width = 64,
            block_function = partial(
                    senet.se_resnext_bottleneck_block,
                    ratio = ratio,
                    regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                    se_kernel_init_1 = He_normal(seed = seed+4),
                    se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = seed+4),
                    ),
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "se_residual_3")
    layers.append(("se_residual_3", res3))

    pool = global_avg_pool2d(res3)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables


#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, dense, global_avg_pool2d
from arch.initializers import He_normal, Kumar_normal
from arch import densenet
from functools import partial


# depth 40 -> 40-4 (initial + 2 trans + 1 dense) = 36
# 36/3 = 12 -> 12 convolutions per block
def cifar10_densenet(x, n_repeat, drop_rate = 0.2, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d(
            x, size = 3, n_filters = 16,
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
    
    dblock1 = densenet.dense_block(
            conv, n_repeat = n_repeat, n_filters = 12,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            seed = seed+2,
            name = "dense_block_1")
    layers.append(("dense_block_1", dblock1))
    
    tlayer1 = densenet.transition_layer(
            dblock1, pool_size = 2, pool_stride = 2,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            seed = seed+3,
            name = "transition_layer_1")
    layers.append(("transition_layer_1", tlayer1))
    
    dblock2 = densenet.dense_block(
            tlayer1, n_repeat = n_repeat, n_filters = 12,
            drop_rate = drop_rate,
            is_training=training,
            kernel_init = He_normal(seed = seed+4),
            seed = seed+4,
            name = "dense_block_2")
    layers.append(("dense_block_2", dblock2))
    
    tlayer2 = densenet.transition_layer(
            dblock2, pool_size = 2, pool_stride = 2,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+5),
            seed = seed+5,
            name = "transition_layer_2")
    layers.append(("transition_layer_2", tlayer2))
    
    dblock3 = densenet.dense_block(
            tlayer2, n_repeat = n_repeat, n_filters = 12,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+6),
            seed = seed+6,
            name = "dense_block_3")
    layers.append(("dense_block_3", dblock3))
    
    final = densenet.final_layer(dblock3, is_training = training, name = "final")
    layers.append(("final", final))

    dense1 = dense(
            final, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+7),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables

# 3*12+4 = 40
cifar10_densenet_40 = partial(cifar10_densenet, n_repeat = 12)


#https://github.com/Lasagne/Recipes/blob/master/papers/densenet/densenet.py
def cifar10_densenet_wd(x, n_repeat, drop_rate = 0.2, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d(
            x, size = 3, n_filters = 16,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
    
    dblock1 = densenet.dense_block(
            conv, n_repeat = n_repeat, n_filters = 12,
            drop_rate = drop_rate,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2),
            seed = seed+2,
            name = "dense_block_1")
    layers.append(("dense_block_1", dblock1))
    
    tlayer1 = densenet.transition_layer(
            dblock1, pool_size = 2, pool_stride = 2,
            drop_rate = drop_rate,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+3),
            seed = seed+3,
            name = "transition_layer_1")
    layers.append(("transition_layer_1", tlayer1))
    
    dblock2 = densenet.dense_block(
            tlayer1, n_repeat = n_repeat, n_filters = 12,
            drop_rate = drop_rate,
            is_training=training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4),
            seed = seed+4,
            name = "dense_block_2")
    layers.append(("dense_block_2", dblock2))
    
    tlayer2 = densenet.transition_layer(
            dblock2, pool_size = 2, pool_stride = 2,
            drop_rate = drop_rate,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+5),
            seed = seed+5,
            name = "transition_layer_2")
    layers.append(("transition_layer_2", tlayer2))
    
    dblock3 = densenet.dense_block(
            tlayer2, n_repeat = n_repeat, n_filters = 12,
            drop_rate = drop_rate,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+6),
            seed = seed+6,
            name = "dense_block_3")
    layers.append(("dense_block_3", dblock3))
    
    final = densenet.final_layer(dblock3, is_training = training, name = "final")
    layers.append(("final", final))

    dense1 = dense(
            final, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+7),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables

cifar10_densenet_40_wd = partial(cifar10_densenet_wd, n_repeat = 12)


# depth 100 -> 100-4 (initial + 2 trans + 1 dense) = 96
# 96/3 = 32 -> 16x2 convolutions per block (16 1x1 + 16 3x3)
def cifar10_bottleneck_densenet(x, n_repeat, drop_rate = 0.25, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d(
            x, size = 3, n_filters = 16,
            kernel_init = He_normal(seed = seed+1),
            name = "initial_conv")
    layers.append(("initial_conv", conv))
    
    dblock1 = densenet.bottleneck_block(
            conv, n_repeat = n_repeat, n_filters = 12, reduction_ratio = 4,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            seed = seed+2,
            name = "dense_bootleneck_block_1")
    layers.append(("dense_bootleneck_block_1", dblock1))
    
    tlayer1 = densenet.transition_layer(
            dblock1, pool_size = 2, pool_stride = 2,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            seed = seed+3,
            name = "transition_layer_1")
    layers.append(("transition_layer_1", tlayer1))
    
    dblock2 = densenet.bottleneck_block(
            tlayer1, n_repeat = n_repeat, n_filters = 12, reduction_ratio = 4,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            seed = seed+4,
            name = "dense_bootleneck_block_2")
    layers.append(("dense_bootleneck_block_2", dblock2))
    
    tlayer2 = densenet.transition_layer(
            dblock2, pool_size = 2, pool_stride = 2,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+5),
            seed = seed+5,
            name = "transition_layer_2")
    layers.append(("transition_layer_2", tlayer2))
    
    dblock3 = densenet.bottleneck_block(
            tlayer2, n_repeat = n_repeat, n_filters = 12, reduction_ratio = 4,
            drop_rate = drop_rate,
            is_training = training,
            kernel_init = He_normal(seed = seed+6),
            seed = seed+6,
            name = "dense_bootleneck_block_3")
    layers.append(("dense_bootleneck_block_3", dblock3))
    
    pool = global_avg_pool2d(dblock3)
    layers.append(("pool", pool))

    dense1 = dense(
            pool, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+7),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables

cifar10_bottleneck_densenet_100 = partial(cifar10_bottleneck_densenet, n_repeat = 16)


#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, Kumar_initializer


def group_conv2d(
        inputs, size, n_filters, split_depth,
        stride = 1,
        activation = tf.nn.relu,
        kernel_init = Kumar_initializer(),
        name = "group_conv2d"):
    
    cardinality = n_filters // split_depth
    
    if cardinality == 1:
        return conv2d(
                inputs, size = size, n_filters = n_filters,
                stride = stride,
                activation = activation,
                kernel_init = kernel_init,
                name = name)
        
    
    with tf.variable_scope(name):
        size_splits = [split_depth]*cardinality
        groups = tf.split(inputs, size_splits, axis=3)
        conv_groups = []
        for i,group in enumerate(groups):
            conv = conv2d(
                    group, size = size, n_filters = split_depth,
                    stride = stride,
                    activation = activation,
                    kernel_init = kernel_init,
                    name = "conv2d_"+str(i))
            conv_groups.append(conv)
            
        outputs = tf.concat(conv_groups, axis=3)    
    return outputs
    

def bottleneck_block(
        inputs,
        n_filters_reduce,
        n_filters,
        split_depth = 4,
        stride = 1,
        activation = tf.nn.relu,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        is_training = False,
        name = "bottleneck_block"):
    with tf.variable_scope(name):        
        if (inputs.shape[3] != n_filters) or (stride != 1):
            shortcut = conv2d(
                        inputs, size=1, n_filters=n_filters,
                        stride = stride, activation = None,
                        kernel_init = kernel_init,
                        name = "shortcut_projection")
        else:
            shortcut = tf.identity(inputs, name="shortcut")
        
        x = conv2d(
                inputs, size=1, n_filters = n_filters_reduce,
                stride = stride,
                activation = None,
                kernel_init = kernel_init,
                name = "conv2d_1")
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_1")
        x = activation(x, name="activation_1")
        

        x = group_conv2d(
                x, size=3, n_filters = n_filters_reduce,
                split_depth = split_depth,
                activation = None,
                kernel_init = kernel_init,
                name = "conv2d_2"
                )
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_2")
        x = activation(x, name="activation_2")

        x = conv2d(
                x, size=1, n_filters = n_filters,
                stride = 1,
                activation = None,
                kernel_init = kernel_init,
                name = "conv2d_3")
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_3")

        x = tf.add(x, shortcut, name="add")
        x = activation(x, name="activation_3")
    return x


def residual_layer(
        inputs,
        n_filters_reduce,
        n_filters,
        split_depth = 4,
        n_blocks = 3, 
        stride = 1,
        is_training = False,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        name = "residual"
        ):
    with tf.variable_scope(name):
        x = bottleneck_block(
                inputs,
                n_filters_reduce = n_filters_reduce,
                n_filters = n_filters,
                split_depth = split_depth,
                stride = stride,
                is_training = is_training,
                kernel_init = kernel_init,
                name = "block_0")
        
        for n in range(1, n_blocks):
            x = bottleneck_block(
                    x,
                    n_filters_reduce = n_filters_reduce,
                    n_filters = n_filters,
                    split_depth = split_depth,
                    stride = 1,
                    is_training = is_training,
                    kernel_init = kernel_init,
                    name = "block_" + str(n+1))
    return x
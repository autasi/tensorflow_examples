#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, Kumar_initializer

## initialization
## https://arxiv.org/pdf/1709.02956.pdf
## batch norm: http://torch.ch/blog/2016/02/04/resnets.html
def residual_block(
        inputs,
        n_filters,
        stride = 1,
        activation = tf.nn.relu,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        is_training = False,
        name = "residual_block"):   
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
                inputs, size=3, n_filters = n_filters,
                stride = stride,
                activation = None,
                kernel_init = kernel_init,
                name = "conv2d_1")
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_1")
        x = activation(x, name="activation_1")
        
        x = conv2d(
                x, size=3, n_filters = n_filters,
                stride = 1,
                activation = None,
                kernel_init = kernel_init,
                name = "conv2d_2")
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_2")

        x = tf.add(x, shortcut, name="add")
        x = activation(x, name="activation_2")
    return x


def bottleneck_block(
        inputs,
        n_filters,
        stride = 1,
        activation = tf.nn.relu,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        is_training = False,
        name = "bottleneck_block"):
    n_filters_reduce = n_filters[0]
    n_filters_conv = n_filters[1]
    with tf.variable_scope(name):        
        if (inputs.shape[3] != n_filters_conv) or (stride != 1):
            shortcut = conv2d(
                        inputs, size=1, n_filters=n_filters_conv,
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
        
        x = conv2d(
                x, size=3, n_filters = n_filters_reduce,
                stride = 1,
                activation = None,
                kernel_init = kernel_init,
                name = "conv2d_2")
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_2")
        x = activation(x, name="activation_2")

        x = conv2d(
                x, size=1, n_filters = n_filters_conv,
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
        n_filters,
        n_blocks = 3, 
        stride = 1,
        block_function = residual_block,
        is_training = False,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        name = "residual"
        ):
    with tf.variable_scope(name):
        x = block_function(inputs, n_filters = n_filters,
                           stride = stride,
                           is_training = is_training,
                           kernel_init = kernel_init,
                           name = "residual_block_0")
        
        for n in range(1, n_blocks):
            x = block_function(x, n_filters = n_filters,
                               stride = 1,
                               is_training = is_training,
                               kernel_init = kernel_init,
                               name = "residual_block_" + str(n+1))
    return x
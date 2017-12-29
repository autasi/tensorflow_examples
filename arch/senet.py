#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, dense, Kumar_initializer
from arch.resnext import group_conv2d


#Squeeze-and-Excitation Networks
#https://arxiv.org/abs/1709.01507
#https://github.com/titu1994/keras-squeeze-excite-network
#https://github.com/hujie-frank/SENet


def squeeze_and_excite(
        inputs,
        ratio = 16,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        name = "squeeze_excite"):
    in_filt = inputs.shape[3].value
    with tf.variable_scope(name):
        x = tf.reduce_mean(inputs, [1,2])        
        x = dense(
                x, n_units=in_filt//ratio,
                activation=tf.nn.relu,
                kernel_init=kernel_init,
                name="fc1")
        x = dense(
                x, n_units=in_filt,
                activation=tf.nn.sigmoid,
                kernel_init=Kumar_initializer(activation="sigmoid", mode="FAN_AVG"),
                name="fc2")
        x = tf.reshape(x, [-1, 1, 1, in_filt])
        outputs = tf.multiply(inputs, x)
    return outputs


def se_resnet_residual_block(
        inputs,
        n_filters,
        stride = 1,
        activation = tf.nn.relu,
        ratio = 16,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        is_training = False,
        name = "se_resnet_residual_block"):   
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

        x = squeeze_and_excite(
                x,
                ratio=ratio,
                kernel_init=kernel_init,
                name="squeeze_excite")
        
        x = tf.add(x, shortcut, name="add")
        x = activation(x, name="activation_2")
    return x


def se_resnext_bottleneck_block(
        inputs,
        n_filters_reduce,
        n_filters,
        split_depth = 4,
        stride = 1,
        ratio = 16,
        activation = tf.nn.relu,
        kernel_init = Kumar_initializer(mode="FAN_IN"),
        is_training = False,
        name = "se_resnext_bottleneck_block"):
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

        x = squeeze_and_excite(
                x,
                ratio=ratio,
                kernel_init=kernel_init,
                name="squeeze_excite")

        x = tf.add(x, shortcut, name="add")
        x = activation(x, name="activation_3")
    return x
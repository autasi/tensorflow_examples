#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, separable_conv2d, Kumar_initializer


#MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
#https://arxiv.org/abs/1704.04861
#https://github.com/Zehaos/MobileNet

def mobilenet_block(
        inputs,
        n_filters,
        stride = 1,
        conv_size = 3,
        alpha = 1.0,
        activation = tf.nn.relu,
        kernel_init = Kumar_initializer(),
        is_training = False,
        name = "mobilenet_block"
        ):
    
    in_filt = inputs.shape[3].value
    n_filters_sep = int(in_filt*alpha)
    n_filters_conv = int(n_filters*alpha)
    with tf.variable_scope(name):
        x = separable_conv2d(
                inputs, size=conv_size, n_filters=n_filters_sep,
                stride=stride, activation=None,
                depth_kernel_init=kernel_init,
                pointwise_kernel_init=kernel_init,
                name="separable_conv")
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_1")
        if activation is not None:
            x = activation(x, name="activation_1")
        x = conv2d(
                x, size=1, n_filters=n_filters_conv,
                stride=1, activation=None,
                kernel_init=kernel_init,
                name="conv")
        x = tf.layers.batch_normalization(x, training=is_training, name="bn_2")
        if activation is not None:
            x = activation(x, name="activation_2")
    return x
        
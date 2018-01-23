#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, separable_conv2d
from arch.initializers import He_normal


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
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        bias_init = tf.zeros_initializer(),
        is_training = False,
        name = "mobilenet_block"
        ):
    
    in_filt = inputs.shape[3].value
    n_filters_sep = int(in_filt*alpha)
    n_filters_conv = int(n_filters*alpha)
    with tf.variable_scope(name):
        x = separable_conv2d(
                inputs, size = conv_size, n_filters = n_filters_sep,
                stride = stride,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv")
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm_1")
        if activation is not None:
            x = activation(x, name = "activation_1")
        x = conv2d(
                x, size = 1, n_filters = n_filters_conv, stride = 1,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv")
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm_2")
        if activation is not None:
            x = activation(x, name = "activation_2")
    return x
        
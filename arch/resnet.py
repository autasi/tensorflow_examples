#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, conv2d_bn_act, conv2d_bn
from arch.initializers import He_normal

# He et al. Deep Residual Learning for Image Recognition, 2015
# https://arxiv.org/pdf/1512.03385.pdf
# initialization
# https://arxiv.org/pdf/1709.02956.pdf
# batch norm: http://torch.ch/blog/2016/02/04/resnets.html
def residual_block(
        inputs,
        n_filters,
        size = 3,
        stride = 1,
        activation = tf.nn.relu,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        is_training = False,
        name = "residual_block"):   
    with tf.variable_scope(name):        
        if (inputs.shape[3] != n_filters) or (stride != 1):
            shortcut = conv2d(
                    inputs, size = 1, n_filters = n_filters, stride = stride,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "shortcut")
        else:
            shortcut = tf.identity(inputs, name = "shortcut")
        
        x = conv2d_bn_act(
                inputs, size = size, n_filters = n_filters, stride = stride,
                activation = activation,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "conv_bn_act_1")
                
        x = conv2d_bn(
                x, size = size, n_filters = n_filters, stride = 1,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "conv_bn_2")

        x = tf.add(x, shortcut, name = "add")
        x = activation(x, name = "activation_2")
    return x


def bottleneck_block(
        inputs,
        n_filters,
        n_filters_reduce,
        size = 3,
        stride = 1,
        activation = tf.nn.relu,
        kernel_init = He_normal(seed = 42),
        is_training = False,
        name = "bottleneck_block"):
    with tf.variable_scope(name):        
        if (inputs.shape[3] != n_filters) or (stride != 1):
            shortcut = conv2d(
                    inputs, size = 1, n_filters = n_filters, stride = stride,
                    kernel_init = kernel_init,
                    name = "shortcut")
        else:
            shortcut = tf.identity(inputs, name = "shortcut")
        
        x = conv2d_bn_act(
                inputs, size = 1, n_filters = n_filters_reduce, stride = stride,
                activation = activation,
                is_training = is_training,
                kernel_init = kernel_init,
                name = "conv_bn_act_1")
        
        x = conv2d_bn_act(
                x, size = size, n_filters = n_filters_reduce, stride = 1,
                activation = activation,
                is_training = is_training,                
                kernel_init = kernel_init,
                name = "conv_bn_act_2")

        x = conv2d_bn(
                x, size = 1, n_filters = n_filters, stride = 1,
                is_training = is_training,
                kernel_init = kernel_init,
                name = "conv_bn_3")

        x = tf.add(x, shortcut, name = "add")
        x = activation(x, name = "activation_3")
    return x


def residual_layer(
        inputs,
        n_filters,
        n_blocks,
        stride = 1,
        block_function = residual_block,
        is_training = False,
        kernel_init = He_normal(seed = 42),
        name = "residual_layer"
        ):
    with tf.variable_scope(name):
        x = block_function(
                inputs, n_filters = n_filters, stride = stride,
                is_training = is_training,
                kernel_init = kernel_init,
                name = "residual_block_0")
        
        for n in range(1, n_blocks):
            x = block_function(
                    x, n_filters = n_filters, stride = 1,
                    is_training = is_training,
                    kernel_init = kernel_init,
                    name = "residual_block_" + str(n))
    return x
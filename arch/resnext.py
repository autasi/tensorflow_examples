#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, conv2d_bn_act, conv2d_bn, group_conv2d_fixdepth
from arch.initializers import He_normal

#Xie et al. Aggregated Residual Transformations for Deep Neural Networks, 2017
#https://arxiv.org/pdf/1611.05431.pdf    
#https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
#https://github.com/titu1994/Keras-ResNeXt/blob/master/resnext.py
def bottleneck_block(
        inputs,
        cardinality,
        group_width,
        size = 3,
        stride = 1,
        activation = tf.nn.relu,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        name = "bottleneck_block"):
    n_filters_reduce = cardinality*group_width
    n_filters = n_filters_reduce*2
    with tf.variable_scope(name):        
        if (inputs.shape[3] != n_filters) or (stride != 1):
            shortcut = conv2d_bn(
                    inputs, size = 1, n_filters = n_filters, stride = stride,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "shortcut")
        else:
            shortcut = tf.identity(inputs, name = "shortcut")
        
        x = conv2d_bn_act(
                inputs, size = 1, n_filters = n_filters_reduce, stride = 1,
                activation = activation,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "conv_bn_act_1")

        x = group_conv2d_fixdepth(
                x, size = size, stride = stride,
                cardinality = cardinality,
                group_width = group_width,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "group_conv_2"
                )
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm_2")
        x = activation(x, name = "activation_2")

        x = conv2d_bn(
                x, size = 1, n_filters = n_filters, stride = 1,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "conv_bn_3")

        x = tf.add(x, shortcut, name = "add")
        x = activation(x, name = "activation_3")
    return x


def residual_layer(
        inputs,
        cardinality,
        group_width,
        n_blocks = 3, 
        stride = 1,
        is_training = False,
        block_function = bottleneck_block,
        kernel_init = He_normal(seed = 42),
        name = "aggregated_residual_layer"
        ):
    with tf.variable_scope(name):
        x = block_function(
                inputs,
                cardinality = cardinality,
                group_width = group_width,
                stride = stride,
                is_training = is_training,
                kernel_init = kernel_init,
                name = "residual_block_0")
        
        for n in range(1, n_blocks):
            x = block_function(
                    x,
                    cardinality = cardinality,
                    group_width = group_width,
                    stride = 1,
                    is_training = is_training,
                    kernel_init = kernel_init,
                    name = "residual_block_" + str(n))
    return x
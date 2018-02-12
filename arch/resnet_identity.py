#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.initializers import He_normal
from arch.layers import conv2d, bn_act_conv2d

#He et al., Identity Mappings in Deep Residual Networks
#https://arxiv.org/abs/1603.05027
#https://github.com/raghakot/keras-resnet
def identity_mapping_block(
        inputs,
        n_filters,
        size = 3,
        stride = 1,
        activation = tf.nn.relu,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        is_training = False,
        skip_first_bn_act = False,
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
        
        if skip_first_bn_act:
            x = conv2d(
                    inputs, size = size, n_filters = n_filters, stride = stride,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1")
        else:
            x = bn_act_conv2d(
                    inputs, size = size, n_filters = n_filters, stride = stride,
                    activation = activation,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "bn_act_conv_1")
                
        x = bn_act_conv2d(
                x, size = size, n_filters = n_filters, stride = stride,
                activation = activation,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "bn_act_conv_2")

        x = tf.add(x, shortcut, name = "add")
    return x
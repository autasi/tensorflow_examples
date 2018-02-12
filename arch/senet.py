#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, conv2d_bn_act, conv2d_bn, group_conv2d_fixdepth
from arch.layers import dense_relu, dense_sigmoid, global_avg_pool2d
from arch.initializers import He_normal, Kumar_normal


#He et al. Squeeze-and-Excitation Networks
#https://arxiv.org/abs/1709.01507
#https://github.com/titu1994/keras-squeeze-excite-network
#https://github.com/hujie-frank/SENet

def squeeze_and_excite(
        inputs,
        ratio = 16,
        regularizer = None,
        kernel_init_1 = He_normal(seed = 42),
        kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = 42),
        name = "squeeze_excite"):
    in_filt = inputs.shape[3].value
    with tf.variable_scope(name):
        x = global_avg_pool2d(inputs)
        x = dense_relu(
                x, n_units = in_filt//ratio,
                regularizer = regularizer,
                kernel_init = kernel_init_1,
                name = "dense_1")
        x = dense_sigmoid(
                x, n_units = in_filt,
                regularizer = regularizer,
                kernel_init = kernel_init_2,
                name = "dense_2")
        x = tf.reshape(x, [-1, 1, 1, in_filt])
        outputs = tf.multiply(inputs, x)
    return outputs


def se_resnet_residual_block(
        inputs,
        n_filters,
        size = 3,
        stride = 1,
        activation = tf.nn.relu,
        ratio = 16,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        se_kernel_init_1 = He_normal(seed = 42),
        se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = 42),
        is_training = False,
        name = "se_resnet_residual_block"):   
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

        x = squeeze_and_excite(
                x, ratio = ratio,
                regularizer = regularizer,
                kernel_init_1 = se_kernel_init_1,
                kernel_init_2 = se_kernel_init_2,
                name = "squeeze_excite")
        
        x = tf.add(x, shortcut, name = "add")
        x = activation(x, name = "activation_2")
    return x


def se_resnet_bottleneck_block(
        inputs,
        n_filters,
        n_filters_reduce,
        size = 3,
        stride = 1,
        activation = tf.nn.relu,
        ratio = 16,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        se_kernel_init_1 = He_normal(seed = 42),
        se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = 42),
        is_training = False,
        name = "se_resnet_residual_block"):   
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
                inputs, size = 1, n_filters = n_filters_reduce, stride = stride,
                activation = activation,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "conv_bn_act_1")
        
        x = conv2d_bn_act(
                x, size = size, n_filters = n_filters_reduce, stride = 1,
                activation = activation,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "conv_bn_act_2")
        
        x = conv2d_bn(
                x, size = size, n_filters = n_filters, stride = 1,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "conv_bn_3")

        x = squeeze_and_excite(
                x, ratio = ratio,
                regularizer = regularizer,
                kernel_init_1 = se_kernel_init_1,
                kernel_init_2 = se_kernel_init_2,
                name = "squeeze_excite")
        
        x = tf.add(x, shortcut, name = "add")
        x = activation(x, name = "activation_3")
    return x


def se_resnext_bottleneck_block(
        inputs,
        cardinality,
        group_width,
        size = 3,
        stride = 1,
        ratio = 16,
        activation = tf.nn.relu,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        se_kernel_init_1 = He_normal(seed = 42),
        se_kernel_init_2 = Kumar_normal(activation = "sigmoid", mode = "FAN_AVG", seed = 42),
        name = "se_resnext_bottleneck_block"):
    n_filters_reduce = cardinality*group_width
    n_filters = n_filters_reduce*2
    with tf.variable_scope(name):        
        if (inputs.shape[3] != n_filters) or (stride != 1):
            shortcut = conv2d_bn(
                    inputs, size = 1, n_filters = n_filters, stride = stride,
                    is_training = is_training,
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

        x = squeeze_and_excite(
                x, ratio = ratio,
                regularizer = regularizer,
                kernel_init_1 = se_kernel_init_1,
                kernel_init_2 = se_kernel_init_2,
                name = "squeeze_excite")

        x = tf.add(x, shortcut, name = "add")
        x = activation(x, name = "activation_3")
    return x
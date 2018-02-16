# -*- coding: utf-8 -*-

import tensorflow as tf

from arch.layers import conv2d_bn_act, separable_conv2d, conv2d_bn
from arch.initializers import He_normal

#Sandler et al., Inverted Residuals and Linear Bottlenecks: Mobile Networks for
#Classification, Detection and Segmentation
#https://arxiv.org/abs/1801.04381v2
#https://github.com/tonylins/pytorch-mobilenet-v2

def inverted_residual(
        inputs,
        n_filters,
        expand_ratio = 1.0,    
        size = 3,
        stride = 1,
        activation = tf.nn.relu6,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        bias_init = tf.zeros_initializer(),
        is_training = False,
        name = "inverted_residual"):

    n_filters_expand = int(n_filters * expand_ratio)
    with tf.variable_scope(name):
        if stride == 1:
            if inputs.shape[3] != n_filters:
                shortcut = conv2d_bn(
                        inputs, size = 1, n_filters = n_filters, stride = 1,
                        is_training = is_training,
                        regularizer = regularizer,
                        kernel_init = kernel_init,
                        bias_init = bias_init,
                        name = "shortcut")
            else:
                shortcut = tf.identity(inputs, name = "shortcut")
        else:
            shortcut = None
            
        # pointwise
        x = conv2d_bn_act(
                inputs, size = 1, n_filters = n_filters_expand, stride = 1,
                activation = activation,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv_1")
                
        # depthwise
        x = separable_conv2d(
                x, size = size, n_filters = n_filters_expand,
                stride = stride,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv")            
        x = tf.layers.batch_normalization(
            x, training = is_training, name = "batch_norm_1")
        if activation is not None:
            x = activation(x, name = "activation")

        # pointwise
        x = conv2d_bn(
                inputs, size = 1, n_filters = n_filters, stride = 1,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv_2")
                
        if shortcut is not None:
            x = tf.add(x, shortcut, name = "add")
    return x


def inverted_residual_block(
        inputs,
        n_filters,
        n_repeat = 2,
        expand_ratio = 1.0,    
        size = 3,
        stride = 1,
        activation = tf.nn.relu6,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        bias_init = tf.zeros_initializer(),
        is_training = False,
        name = "inverted_residual_block"):
        
    with tf.variable_scope(name):
        x = inverted_residual(
                inputs,
                n_filters = n_filters,
                expand_ratio = expand_ratio,
                size = size,
                stride = stride,
                activation = activation,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                is_training = is_training,
                name = "inverted_residual_0")
        
        for n in range(0, n_repeat-1):
            x = inverted_residual(
                    x,
                    n_filters = n_filters,
                    expand_ratio = expand_ratio,
                    size = size,
                    stride = 1,
                    activation = activation,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    is_training = is_training,
                    name = "inverted_residual_"+str(n+1))
    return x        

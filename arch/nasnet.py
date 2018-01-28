#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.layers import conv2d, conv2d_bn, separable_conv2d, avg_pool2d, max_pool2d, zero_pad2d, crop2d, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal

#Learning Transferable Architectures for Scalable Image Recognition
#https://arxiv.org/pdf/1707.07012.pdf
#https://github.com/titu1994/Keras-NASNet
#https://github.com/johannesu/NASNet-keras/blob/master/nasnet.py


#def adjust(p, ip, filters, weight_decay=5e-5, id=None):
def adjust(
    p,
    ref,
    n_filters,
    activation = tf.nn.relu,
    is_training = False,
    regularizer = None,
    kernel_init = He_normal(seed = 42),
    bias_init = tf.zeros_initializer(),
    name = "adjust"):
    
    #NHWC
    channel_dim = 3 
    img_dim = 1

    with tf.variable_scope(name):
        if p is None:
            p = ref

        elif p.get_shape()[img_dim].value != ref.get_shape()[img_dim].value:
            with tf.variable_scope("reduction"):
                p = activation(p, name = "activation")

                p1 = avg_pool2d(
                    p, size = 1, stride = 2,
                    padding = "VALID", name = "avg_pool_1")
                p1 = conv2d(
                    p1, n_filters = n_filters // 2, size = 1, 
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name = "conv_1")

                p2 = zero_pad2d(p, pad = [[0,1], [0,1]], name = "zero_pad")
                p2 = crop2d(p2, crop = [[1,0], [1,0]], name = "crop")
                p2 = avg_pool2d(
                    p2, size = 1, stride = 2,
                    padding = "VALID", name = "avg_pool_2")
                p2 = conv2d(
                    p2, n_filters = n_filters // 2, size = 1, 
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name = "conv_2")

                p = tf.concat([p1, p2], axis = 3, name = "concat")
                p = tf.layers.batch_normalization(
                    p, training = is_training, name = "batch_norm")                

        elif p.get_shape()[channel_dim].value != n_filters:
            with tf.variable_scope("projection"):
                p = activation(p, name = "activation")
                p = conv2d_bn(
                    p, n_filters = n_filters, size = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name = "conv_bn")                
    return p


def squeeze(
    x,
    n_filters,
    activation = tf.nn.relu,
    is_training = False,
    regularizer = None,
    kernel_init = He_normal(seed = 42),
    bias_init = tf.zeros_initializer(),
    name = "squeeze"):
        
    with tf.variable_scope(name):
        if activation is not None:
            x = activation(x, name = "activation")
        x = conv2d_bn(
            x, n_filters = n_filters, size = 1, stride = 1,
            is_training = is_training,
            regularizer = regularizer,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name = "conv")
    return x


def Normal_A(
    inputs, p,
    n_filters,
    activation = tf.nn.relu,
    is_training = False,
    regularizer = None,
    kernel_init = He_normal(seed = 42),
    bias_init = tf.zeros_initializer(),
    name = "nasnet_normal_a"
    ):

    with tf.variable_scope(name):
        # adjust to the projection
        p = adjust(p, ref = inputs, n_filters = n_filters, name = "adjust")

        # squeeze inputs to match dimensions
        h = squeeze(
            inputs, n_filters = n_filters,
            is_training = is_training,
            regularizer = regularizer,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name = "squeeze")

        # block 1
        with tf.variable_scope("block_1"):
            x1 = separable_conv2d(
                h, size = 3, n_filters = n_filters, stride = 1,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_1")
            x1 = tf.add(x1, h, name = "add_1")

        # block 2
        with tf.variable_scope("block_2"):
            x2_1 = separable_conv2d(
                h, size = 5, n_filters = n_filters, stride = 1,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_2_h")
            x2_2 = separable_conv2d(
                p, size = 3, n_filters = n_filters, stride = 1,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_2_p")
            x2 = tf.add(x2_1, x2_2, name = "add_2")

        # block 3
        with tf.variable_scope("block_3"):
            x3 = avg_pool2d(h, size = 3, stride = 1, name = "avg_pool_3")
            x3 = tf.add(x3, p, name = "add_3")

        # block 4
        with tf.variable_scope("block_4"):
            x4_1 = avg_pool2d(h, size = 3, stride = 1, name = "avg_pool_4_1")
            x4_2 = avg_pool2d(h, size = 3, stride = 1, name = "avg_pool_4_2")
            x4 = tf.add(x4_1, x4_2, name = "add_4")

        # block 5
        with tf.variable_scope("block_5"):
            x5_1 = separable_conv2d(
                p, size = 5, n_filters = n_filters, stride = 1,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_5_5x5")
            x5_2 = separable_conv2d(
                p, size = 3, n_filters = n_filters, stride = 1,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_5_3x3")
            x5 = tf.add(x5_1, x5_2, name = "add_5")

        outputs = tf.concat([p, x1, x2, x3, x4, x5], axis = 3, name = "concat")
    return outputs, inputs
    
    

def Reduction_A(
    inputs, p,
    n_filters,
    is_training = False,
    regularizer = None,
    activation = tf.nn.relu,
    kernel_init = He_normal(seed = 42),
    bias_init = tf.zeros_initializer(),
    name = "nasnet_normal_a"
    ):

    with tf.variable_scope(name):
        # adjust to the reduction
        p = adjust(p, ref = inputs, n_filters = n_filters, name = "adjust")

        # squeeze inputs to match dimensions
        h = squeeze(
            inputs, n_filters = n_filters,
            is_training = is_training,
            regularizer = regularizer,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name = "squeeze")

        with tf.variable_scope("block_1"):
            x1_1 = separable_conv2d(
                h, size = 5, n_filters = n_filters, stride = 2,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_1_5x5")
            x1_2 = separable_conv2d(
                p, size = 7, n_filters = n_filters, stride = 2,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_1_7x7")
            x1 = tf.add(x1_1, x1_2, name = "add_1")

        with tf.variable_scope("block_2"):
            x2_1 = max_pool2d(h, size = 3, stride = 2, name = "max_pool_2")
            x2_2 = separable_conv2d(
                p, size = 7, n_filters = n_filters, stride = 2,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_2")
            x2 = tf.add(x2_1, x2_2, name = "add_2")

        with tf.variable_scope("block_3"):
            x3_1 = avg_pool2d(h, size = 3, stride = 2, name = "avg_pool_3")
            x3_2 = separable_conv2d(
                p, size = 5, n_filters = n_filters, stride = 2,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_3")
            x3 = tf.add(x3_1, x3_2, name = "add_3")

        with tf.variable_scope("block_4"):
            x4_1 = max_pool2d(h, size = 3, stride = 2, name = "max_pool_4")
            x4_2 = separable_conv2d(
                x1, size = 3, n_filters = n_filters, stride = 1,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv_4")
            x4 = tf.add(x4_1, x4_2, name = "add_4")

        with tf.variable_scope("block_5"):
            x5 = avg_pool2d(x1, size = 3, stride = 1, name = "avg_pool_5")
            x5 = tf.add(x2, x5, name = "add_5")

        outputs = tf.concat([x2, x3, x4, x5], axis = 3, name = "concat")
    return outputs, inputs


def auxiliary_classifier(
    inputs, classes,
    is_training = False,
    regularizer = None,
    activation = tf.nn.relu,
    conv_kernel_init = He_normal(seed = 42),
    conv_bias_init = tf.zeros_initializer(),
    dense_kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = 42),
    name = "nasnet_auxiliary_classifier"):
    
    with tf.variable_scope(name):
        x = inputs
        if activation is not None:
            x = activation(x, name = "activation_1")
        x = avg_pool2d(x, size = 5, stride = 3, padding = "VALID", name = "avg_pool")
        
        x = conv2d_bn(
            x, n_filters = 128, size = 1, 
            is_training = is_training,
            regularizer = regularizer,
            kernel_init = conv_kernel_init,
            bias_init = conv_bias_init,
            name = "conv_projection")
        
        if activation is not None:
            x = activation(x, name = "activation_2")
        
        x = conv2d_bn(
            x, n_filters = 768, size = [x.get_shape()[1].value, x.get_shape()[2].value],
            padding = "VALID",
            is_training = is_training,
            regularizer = regularizer,
            kernel_init = conv_kernel_init,
            bias_init = conv_bias_init,
            name = "conv_reduction")

        if activation is not None:
            x = activation(x, name = "activation_3")
            
        x = global_avg_pool2d(x, name = "global_avg_pool")

        x = dense(
                x, n_units = classes,
                regularizer = regularizer,
                kernel_init = dense_kernel_init,
                name = "dense")

    return x
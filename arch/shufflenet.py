#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.layers import separable_conv2d, group_conv2d, avg_pool2d
from arch.initializers import He_normal

#Zhang et al., ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
#https://arxiv.org/pdf/1707.01083.pdf
#https://github.com/MG2033/ShuffleNet
#https://github.com/TropComplique/ShuffleNet-tensorflow
#https://github.com/jinyu121/Tensorflow-ShuffleNet

def shuffle_channels(
        inputs,
        n_groups,
        name = "shuffle"):
    with tf.variable_scope(name):
        h, w, n_chans = inputs.get_shape().as_list()[1:]
        chans_per_group = n_chans // n_groups
        x = tf.reshape(inputs, [-1, h, w, n_groups, chans_per_group])
        x = tf.transpose(x, perm=[0,1,2,4,3])
        x = tf.reshape(x, [-1, h, w, n_chans])
    return x
        

def shuffle_unit(
        inputs,
        n_filters,
        size = 3,
        stride = 1,
        activation = tf.nn.relu,
        reduction_ratio = 0.25,
        n_groups = 8,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        bias_init = tf.zeros_initializer(),
        name = "shuffle_unit"):
    with tf.variable_scope(name):
        n_filters_reduction = int(n_filters*reduction_ratio)
        
        if stride == 1:
            shortcut = tf.identity(inputs, name = "shortcut")
        else:
            shortcut = avg_pool2d(
                inputs, size = 3, stride = stride, name = "shortcut_pool")

        x = group_conv2d(
                inputs, size = 1, cardinality = n_groups,
                n_filters = n_filters_reduction,
                stride = 1,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "group_conv2d_1")
        x = tf.layers.batch_normalization(
            x, training = is_training, name = "batch_norm_1")
        x = activation(x, name="activation_1")
        
        x = shuffle_channels(x, n_groups = n_groups, name = "shuffle")
    
        x = separable_conv2d(
                x, size = size, n_filters = n_filters,
                stride = stride,
                regularizer = regularizer,
                depth_init = kernel_init,
                pointwise_init = kernel_init,
                bias_init = bias_init,
                name = "separable_conv")    
        x = tf.layers.batch_normalization(
            x, training = is_training, name = "batch_norm_2")

        x = group_conv2d(
                x, size = 1, cardinality = n_groups,
                n_filters = n_filters if stride == 1 else n_filters-inputs.shape[3].value,
                stride = 1,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "group_conv2d_2")
        x = tf.layers.batch_normalization(
            x, training = is_training, name="batch_norm_3")
    
        if stride == 1:
            x = tf.add(x, shortcut, name = "add")
        else:
            x = tf.concat([x, shortcut], axis = 3, name = "concat")
        x = activation(x, name = "activation_2")
        
    return x


def shufflenet_layer(
        inputs,
        n_filters,
        n_repeat,
        size = 3,
        reduction_ratio = 0.25,
        n_groups = 8,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        bias_init = tf.zeros_initializer(),
        name = "shufflenet_layer"
        ):
    with tf.variable_scope(name):
        x = shuffle_unit(
                inputs,
                n_filters = n_filters,
                size = size,
                stride = 2,
                activation = tf.nn.relu,
                reduction_ratio = reduction_ratio,
                n_groups = n_groups,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                is_training = is_training,
                name = "shuffle_unit_0")
        
        for n in range(0, n_repeat):
            x = shuffle_unit(
                    x,
                    n_filters = n_filters,
                    size = size,
                    stride = 1,
                    activation = tf.nn.relu,
                    reduction_ratio = reduction_ratio,
                    n_groups = n_groups,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    is_training = is_training,
                    name = "shuffle_unit_"+str(n+1))
    return x


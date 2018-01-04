#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.layers import bn_act_conv2d, avg_pool2d, global_avg_pool2d
from arch.initializers import He_normal

#Densely Connected Convolutional Networks
#https://arxiv.org/abs/1608.06993
#https://github.com/YixuanLi/densenet-tensorflow


def dense_block(
        inputs,
        n_repeat = 5,
        n_filters = 12,
        size = 3,
        drop_rate = 0.2,
        activation = tf.nn.relu,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        is_training = False,
        seed = 42,
        name = "dense_block"):
    with tf.variable_scope(name):    
        shortcuts = []
        shortcuts.append(inputs)
        x = inputs
        for r in range(n_repeat):
            x = bn_act_conv2d(
                    x, size = size, n_filters = n_filters, stride = 1,
                    activation = activation,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "bn_act_conv_" + str(r))
            x = tf.layers.dropout(
                    x, rate = drop_rate,
                    training = is_training,
                    seed = seed, name = "dropout_"+str(r))
            shortcuts.append(x)
            x = tf.concat(shortcuts, axis = 3)
    return x


def bottleneck_block(
        inputs,
        n_repeat = 5,
        n_filters = 12,
        size = 3,
        drop_rate = 0.2,
        activation = tf.nn.relu,
        reduction_ratio = 4,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        is_training = False,
        seed = 42,
        name = "dense_bottleneck_block"):
    n_filters_reduction = n_filters*reduction_ratio
    with tf.variable_scope(name):    
        shortcuts = []
        shortcuts.append(inputs)
        x = inputs
        for r in range(n_repeat):
            x = bn_act_conv2d(
                    x, size = 1, n_filters = n_filters_reduction, stride = 1,
                    activation = activation,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "reduction_bn_act_conv_" + str(r))
            
            x = tf.layers.dropout(
                    x, rate = drop_rate,
                    training = is_training,
                    seed = seed, name = "reduction_dropout_"+str(r))
            
            x = bn_act_conv2d(
                    x, size = size, n_filters = n_filters, stride = 1,
                    activation = activation,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "bn_act_conv_" + str(r))
            
            x = tf.layers.dropout(
                    x, rate = drop_rate,
                    training = is_training,
                    seed = seed, name = "dropout_"+str(r))
            
            shortcuts.append(x)
            x = tf.concat(shortcuts, axis = 3)
    return x


def transition_layer(
        inputs,
        pool = avg_pool2d,
        pool_size = 2,
        pool_stride = 2,
        drop_rate = 0.2,
        theta = 1.0,
        activation = tf.nn.relu,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        seed = 42,
        name = "trasition_layer"):
    in_filt = inputs.shape[3].value
    n_filters = int(in_filt*theta)
    with tf.variable_scope(name):
        x = bn_act_conv2d(
                inputs, size = 1, n_filters = n_filters, stride = 1,
                activation = activation,
                is_training = is_training,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = "bn_act_conv")
        
        x = tf.layers.dropout(
                x, rate = drop_rate,
                training = is_training,
                seed = seed, name = "dropout")
        x = pool(x, size = pool_size, stride = pool_stride, name = "pool")
    return x
        

# not used in the original paper: bn+relu+global_avg_pool
def final_layer(
        inputs,
        activation = tf.nn.relu,
        is_training = False,
        name = "final_layer"):
    with tf.variable_scope(name):  
        x = tf.layers.batch_normalization(
                inputs, training = is_training, name = "batch_norm")
        x = activation(x, name = "activation")
        x = global_avg_pool2d(x)
    return x
#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.layers import conv2d, avg_pool
from arch.layers import Kumar_initializer

#Densely Connected Convolutional Networks
#https://arxiv.org/abs/1608.06993
#https://github.com/YixuanLi/densenet-tensorflow


def dense_block(
        inputs,
        n_repeat = 5,
        n_filters = 12,
        conv_size = 3,
        drop_rate = 0.5,
        kernel_init = Kumar_initializer(),
        is_training = False,
        seed = 42,
        name = "dense_block"):
    with tf.variable_scope(name):    
        shortcuts = []
        shortcuts.append(inputs)
        x = inputs
        for r in range(n_repeat):
            x = tf.layers.batch_normalization(x, training=is_training, name="bn_"+str(r))
            x = tf.nn.relu(x, name="relu_"+str(r))
            x = conv2d(
                    x, size=conv_size, n_filters=n_filters,
                    stride=1, activation=None,
                    kernel_init=kernel_init,
                    name="conv_"+str(r))
            x = tf.layers.dropout(x, rate=drop_rate, training=is_training, seed=seed, name="dropout_"+str(r))
            shortcuts.append(x)
            x = tf.concat(shortcuts, axis=3)
    return x



def bottleneck_block(
        inputs,
        n_repeat = 5,
        n_filters = 12,
        conv_size = 3,
        drop_rate = 0.5,
        reduction_ratio = 4,
        kernel_init = Kumar_initializer(),
        is_training = False,
        seed = 42,
        name = "dense_block"):
    n_filters_reduction = n_filters*reduction_ratio
    with tf.variable_scope(name):    
        shortcuts = []
        shortcuts.append(inputs)
        x = inputs
        for r in range(n_repeat):
            x = tf.layers.batch_normalization(x, training=is_training, name="reduction_bn_"+str(r))
            x = tf.nn.relu(x, name="reduction_relu_"+str(r))
            x = conv2d(
                    x, size=1, n_filters=n_filters_reduction,
                    stride=1, activation=None,
                    kernel_init=kernel_init,
                    name="reduction_conv_"+str(r))
            
            x = tf.layers.batch_normalization(x, training=is_training, name="bn_"+str(r))
            x = tf.nn.relu(x, name="relu_"+str(r))
            x = conv2d(
                    x, size=conv_size, n_filters=n_filters,
                    stride=1, activation=None,
                    kernel_init=kernel_init,
                    name="conv_"+str(r))
            x = tf.layers.dropout(x, rate=drop_rate, training=is_training, seed=seed, name="dropout_"+str(r))
            shortcuts.append(x)
            x = tf.concat(shortcuts, axis=3)
    return x


def transition_layer(
        inputs,
        pool = avg_pool,
        pool_size = 2,
        pool_stride = 2,
        drop_rate = 0.5,
        is_training = False,
        kernel_init = Kumar_initializer(),
        seed = 42,
        name = "trasition_layer"):
    in_filt = inputs.shape[3].value
    with tf.variable_scope(name):  
        x = tf.layers.batch_normalization(inputs, training=is_training, name="bn")
        x = tf.nn.relu(x, name="relu")
        x = conv2d(
                x, size=1, n_filters=in_filt,
                stride=1, activation=None,
                kernel_init=kernel_init,
                name="conv_transition")
        x = tf.layers.dropout(x, rate=drop_rate, training=is_training, seed=seed, name="dropout_transition")
        x = pool(x, size=pool_size, stride=pool_stride, name="pool")
    return x
        
# not used in the original paper: bn+relu+global_avg_pool
def final_layer(
        inputs,
        is_training = False,
        name = "final_layer"):
    with tf.variable_scope(name):  
        x = tf.layers.batch_normalization(inputs, training=is_training, name="bn")
        x = tf.nn.relu(x, name="relu")
        x = tf.reduce_mean(x, [1,2])
    return x
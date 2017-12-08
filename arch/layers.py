#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from functools import partial



glorot_u_seeded = partial(tf.glorot_uniform_initializer, seed=42)
"""The seeded version of the Glorot uniform initializer.
"""    


truncated_n_seeded = partial(tf.truncated_normal_initializer, seed=42)
"""The seeded version of the truncated normal initializer.
"""    


def conv2d(inputs, size, n_filters,
           strides = [1,1,1,1],
           activation = tf.nn.relu,
           kernel_init = glorot_u_seeded(),
           bias_init = tf.zeros_initializer(),
           name = "conv2d"):
    """Creates a 2d convolutional layer.
    Args:
        inputs: A tensor representing the inputs.
        size: An integer representing the kernel size.
        n_filters: An integer representing the number of filters.
        strides: A list representing the stride size.
        activation: An activation function to be applied.
        kernel_init: A function used for initializing the kernel.
        bias_init: A function used for initializing the bias.
        name: A string representing the name of the layer.
    Returns:
        A tensor representing the layer.
    """    
    in_filt = inputs.shape[3].value    
    with tf.variable_scope(name):
        weights = tf.get_variable(shape=[size,size,in_filt,n_filters], initializer=kernel_init, name="weight")
        biases = tf.get_variable(shape=[n_filters], initializer=bias_init, name="bias")
        conv = tf.nn.conv2d(inputs, weights, strides=strides, padding="SAME", name="conv")
        bias_add = tf.nn.bias_add(conv, biases, name="bias_add")
        if activation is None:
            outputs = bias_add
        else:
            outputs = activation(bias_add, name="activation")
    return outputs


max_pool = partial(tf.nn.max_pool, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
"""Max-pooling with 2x2 spatial window size, and 2x2 strides.
"""    


def flatten(inputs, name="flatten"):
    """Creates a flattening layer.
    Args:
        inputs: A tensor representing the inputs.
        name: A string representing the name of the layer.
    Returns:
        A tensor representing the layer.
    """    
    return tf.reshape(inputs, [-1, np.prod(inputs.get_shape()[1:].as_list())], name=name)


def dense(inputs, n_units,
          activation = tf.nn.relu,
          kernel_init = truncated_n_seeded(),
          bias_init = tf.zeros_initializer(),
          name = "dense"):
    """Creates a fully-connected dense layer.
    Args:
        inputs: A tensor representing the inputs.
        n_units: An integer representing the number of units.
        activation: An activation function to be applied.
        kernel_init: A function used for initializing the kernel.
        bias_init: A function used for initializing the bias.
        name: A string representing the name of the layer.
    Returns:
        A tensor representing the layer.
    """    
    with tf.variable_scope(name):
        weights = tf.get_variable(shape=[inputs.shape[1].value, n_units], initializer=kernel_init, name="weight")
        biases = tf.get_variable(shape=[n_units], initializer=bias_init, name="bias")
        fc = tf.matmul(inputs, weights, name="matmul")
        bias_add = tf.nn.bias_add(fc, biases, name="bias_add")
        if activation is None:
            outputs = bias_add
        else:
            outputs = activation(bias_add, name="activation")
    return outputs
    

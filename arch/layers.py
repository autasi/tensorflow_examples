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


variance_s_seeded = partial(tf.variance_scaling_initializer, seed=42)
"""The seeded version of the variance scaling initializer.
""" 


def batch_norm_activation(inputs, activation=tf.nn.relu,
                          bn_decay = 0.997,
                          bn_eps =  1e-5,
                          is_training = False):
    bn = tf.layers.batch_normalization(
            inputs = inputs, axis = -1,
            momentum = bn_decay, epsilon = bn_eps,
            center = True, scale = True,
            training = is_training, fused = None)
    if activation is None:
        outputs = bn
    else:
        outputs = activation(bn)
    return outputs


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


def conv2d_bn(inputs, size, n_filters,
              strides = [1,1,1,1],
              activation = tf.nn.relu,
              kernel_init = glorot_u_seeded(),
              bias_init = tf.zeros_initializer(),
              bn_decay = 0.997,
              bn_eps =  1e-5,
              is_training = False,
              name = "conv2d"):
    """Creates a 2d convolutional layer with batch normalization.
    Args:
        inputs: A tensor representing the inputs.
        size: An integer representing the kernel size.
        n_filters: An integer representing the number of filters.
        strides: A list representing the stride size.
        activation: An activation function to be applied.
        kernel_init: A function used for initializing the kernel.
        bias_init: A function used for initializing the bias.
        bn_decay: A number representing the batch normalization decay for the
            moving average.
        bn_eps: A number representing the batch normalization epsilon that is
            used to avoid division by zero.
        is_training: A boolean or a TensorFlow boolean scalar tensor for
            indicating training or testing mode.
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
        bn = tf.layers.batch_normalization(inputs=bias_add, axis=3,
                                           momentum=bn_decay, epsilon=bn_eps,
                                           center=True, scale=True,
                                           training=is_training, fused=None,
                                           name="batch_norm")        
        if activation is None:
            outputs = bn
        else:
            outputs = activation(bn, name="activation")
    return outputs


max_pool = partial(tf.nn.max_pool, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
"""Max-pooling with 2x2 spatial window size, and 2x2 strides.
"""    


avg_pool = partial(tf.nn.avg_pool, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
"""Average-pooling with 2x2 spatial window size, and 2x2 strides.
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
    

def dense_bn(inputs, n_units,
             activation = tf.nn.relu,
             kernel_init = truncated_n_seeded(),
             bias_init = tf.zeros_initializer(),
             bn_decay = 0.997,
             bn_eps =  1e-5,
             is_training = False,
             name = "dense"):
    """Creates a fully-connected dense layer with batch normalization.
    Args:
        inputs: A tensor representing the inputs.
        n_units: An integer representing the number of units.
        activation: An activation function to be applied.
        kernel_init: A function used for initializing the kernel.
        bias_init: A function used for initializing the bias.
        bn_decay: A number representing the batch normalization decay for the
            moving average.
        bn_eps: A number representing the batch normalization epsilon that is
            used to avoid division by zero.
        is_training: A boolean or a TensorFlow boolean scalar tensor for
            indicating training or testing mode.
        name: A string representing the name of the layer.
    Returns:
        A tensor representing the layer.
    """    
    with tf.variable_scope(name):
        weights = tf.get_variable(shape=[inputs.shape[1].value, n_units], initializer=kernel_init, name="weight")
        biases = tf.get_variable(shape=[n_units], initializer=bias_init, name="bias")
        fc = tf.matmul(inputs, weights, name="matmul")
        bias_add = tf.nn.bias_add(fc, biases, name="bias_add")
        bn = tf.layers.batch_normalization(inputs=bias_add, axis=-1,
                                           momentum=bn_decay, epsilon=bn_eps,
                                           center=True, scale=True,
                                           training=is_training, fused=None,
                                           name="batch_norm")
        if activation is None:
            outputs = bn
        else:
            outputs = activation(bn, name="activation")
    return outputs


def residual_block(inputs, n_filters,
                   strides = [1,1,1,1],
                   activation = tf.nn.relu,
                   kernel_init = variance_s_seeded(),
                   bias_init = tf.zeros_initializer(),
                   bn_decay = 0.997,
                   bn_eps =  1e-5,
                   is_training = False,
                   name = "residual_block"):   
    with tf.variable_scope(name):        
        if (inputs.shape[3] != n_filters) or (strides[1:3] != [1,1]):
            shortcut = conv2d(
                        inputs, size=1, n_filters=n_filters,
                        strides = strides, activation = None,
                        kernel_init = variance_s_seeded(),
                        name = "shortcut_projection")
        else:
            shortcut = tf.identity(inputs, name="shortcut")
        
        x = conv2d(
                inputs, size=3, n_filters = n_filters,
                strides = strides,
                activation = None,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv2d_1")

        x = batch_norm_activation(x, activation = activation,
                                  is_training = is_training,
                                  bn_decay = bn_decay,
                                  bn_eps = bn_eps)
        
        x = conv2d(
                x, size=3, n_filters = n_filters,
                strides = [1,1,1,1],
                activation = None,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv2d_2")

        x = batch_norm_activation(x, activation = None,
                                  is_training = is_training,
                                  bn_decay = bn_decay,
                                  bn_eps = bn_eps)
                        
        x = tf.add(x, shortcut, name="add")
        x = activation(x, name="activation")
    return x


def residual_layer(inputs, n_filters, n_blocks, 
                   strides = [1,1,1,1],
                   kernel_init = variance_s_seeded(),
                   is_training = False,
                   name = "residual_layer"
                   ):
    with tf.variable_scope(name):
        x = residual_block(inputs, n_filters = n_filters,
                           strides = strides,
                           kernel_init = kernel_init,
                           is_training = is_training,
                           name = "residual_block_0")
        
        for n in range(1, n_blocks):
            x = residual_block(x, n_filters = n_filters,
                               strides = [1,1,1,1],
                               kernel_init = kernel_init,
                               is_training = is_training,
                               name = "residual_block_" + str(n+1))
    return x

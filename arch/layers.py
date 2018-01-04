#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import tensorflow as tf
from arch.initializers import He_normal

def conv2d(
        inputs, size, n_filters,
        stride = 1,
        padding = "SAME",
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = He_normal(),
        name = "conv2d"):
    """Creates a 2D convolutional layer.
    Args:
        inputs: 4D input tensor, NHWC
        size: Kernel size, int or list of two ints.
        n_filters: Number of filters.
        stride: Stride size, int or list of two ints.
        padding: Padding algorithm "SAME" or "VALID".
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        name: Name of the layer.
    Returns:
        4D tensor.
    """    
    in_filt = inputs.shape[3].value
    if not isinstance(size, (tuple, list)):
        size = [size, size]
    if not isinstance(stride, (tuple, list)):
        stride = [stride, stride]
    with tf.variable_scope(name):
        weights = tf.get_variable(
                shape = [size[0], size[1], in_filt, n_filters],
                regularizer = regularizer,
                initializer = kernel_init,
                name = "weight")
        biases = tf.get_variable(
                shape = [n_filters],
                initializer = bias_init,
                name = "bias")
        conv = tf.nn.conv2d(
                inputs, weights,
                strides = [1, stride[0], stride[1], 1],
                padding = padding,
                name = "conv")
        outputs = tf.nn.bias_add(conv, biases, name = "bias_add")
    return outputs


def conv2d_bn(
        inputs, size, n_filters,
        stride = 1,
        padding = "SAME",
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = He_normal(),
        name = "conv2d_bn"
        ):
    with tf.variable_scope(name):
        x = conv2d(
                inputs, size, n_filters,
                stride = stride,
                padding = padding,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv2d")
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm")
    return x


def conv2d_act(
        inputs, size, n_filters, activation,
        stride = 1,
        padding = "SAME",
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = He_normal(),        
        name = "conv2d_act"
        ):
    with tf.variable_scope(name):
        x = conv2d(
                inputs, size, n_filters,
                stride = stride,
                padding = padding,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv2d")
        if activation is not None:
            x = activation(x, name = "activation")
    return x

conv2d_relu = partial(conv2d_act, activation = tf.nn.relu)

def conv2d_bn_act(
        inputs, size, n_filters, activation,
        stride = 1,
        padding = "SAME",
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = He_normal(),        
        name = "conv2d_bn_act"
        ):
    with tf.variable_scope(name):
        x = conv2d(
                inputs, size, n_filters,
                stride = stride,
                padding = padding,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv2d")
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm")
        if activation is not None:
            x = activation(x, name = "activation")
    return x

conv2d_bn_relu = partial(conv2d_bn_act, activation = tf.nn.relu)


def conv2d_act_bn(
        inputs, size, n_filters, activation,
        stride = 1,
        padding = "SAME",
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = He_normal(),        
        name = "conv2d_bn_act"
        ):
    with tf.variable_scope(name):
        x = conv2d(
                inputs, size, n_filters,
                stride = stride,
                padding = padding,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv2d")
        if activation is not None:
            x = activation(x, name = "activation")
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm")
    return x

conv2d_relu_bn = partial(conv2d_act_bn, activation = tf.nn.relu)


def bn_act_conv2d(
        inputs, size, n_filters, activation,
        stride = 1,
        padding = "SAME",
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = He_normal(),        
        name = "bn_act_conv2d"
        ):
    with tf.variable_scope(name):
        x = tf.layers.batch_normalization(
                inputs, training = is_training, name = "batch_norm")
        if activation is not None:
            x = activation(x, name = "activation")
        x = conv2d(
                x, size, n_filters,
                stride = stride,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "conv2d")
    return x

bn_relu_conv2d = partial(bn_act_conv2d, activation = tf.nn.relu)


def separable_conv2d(
        inputs, size, n_filters,
        stride = 1,
        depth_multiplier = 1,
        padding = "SAME",
        depth_init = He_normal(),
        pointwise_init = He_normal(),
        bias_init = tf.zeros_initializer(),
        name = "separable_conv2d"):
    """Creates a depthwise separable 2D convolutional layer.
    Args:
        inputs: 4D input tensor, NHWC
        size: Kernel size, int or list of two ints.
        n_filters: Number of filters.
        stride: Stride size, int or list of two ints.
        depth_multiplier: Number of depthwise convolution output channels for 
            each input channel.
        padding: Padding algorithm "SAME" or "VALID".
        depth_init: Depthwise initialization function.
        pointwise_init: Pointwise initialization function.
        bias_init: Bias initialization function.
        name: Name of the layer.
    Returns:
        4D tensor.
    """    
    in_filt = inputs.shape[3].value    
    if not isinstance(size, (tuple, list)):
        size = [size, size]
    if not isinstance(stride, (tuple, list)):
        stride = [stride, stride]
    with tf.variable_scope(name):
        depth_weights = tf.get_variable(
                shape = [size[0], size[1], in_filt, depth_multiplier],
                initializer = depth_init,
                name = "depth_weight")
        pointwise_weights = tf.get_variable(
                shape = [1, 1, depth_multiplier*in_filt, n_filters],
                initializer = pointwise_init,
                name = "pointwise_weight")
        biases = tf.get_variable(
                shape = [n_filters], initializer = bias_init, name = "bias")
        conv = tf.nn.separable_conv2d(
                inputs, depth_weights, pointwise_weights,
                strides = [1, stride[0], stride[1], 1],
                padding = padding,
                name = "separable_conv")
        outputs = tf.nn.bias_add(conv, biases, name="bias_add")
    return outputs


def factorized_conv2d(
        inputs, size, n_filters,
        n_repeat = 1,
        stride = 1,
        padding = "SAME",
        kernel_init = He_normal(),
        bias_init = tf.zeros_initializer(),
        name = "factorized_conv2d"):
    """Creates a factorized 2D convolutional layer, i.e. nxn is factorized into
       1xn followed by nx1, which can be repeated multiple times.
    Args:
        inputs: 4D input tensor, NHWC
        size: Kernel size, int.
        n_filters: List of ints or two-tuples containing the number of filters.
        n_repeat: Number of repetitions.
        stride: Stride size, int or list of two ints.
        padding: Padding algorithm "SAME" or "VALID".
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        name: Name of the layer.
    Returns:
        4D tensor.
    """    
    if not isinstance(stride, (tuple, list)):
        stride = [stride, stride]
    if not isinstance(n_filters, list):
        n_filters = [n_filters]*n_repeat
    with tf.variable_scope(name):
        x = inputs
        for r in range(n_repeat):
            curr_filters = n_filters[r]
            if not isinstance(curr_filters, (tuple, list)):
                curr_filters = (curr_filters, curr_filters)
            in_filt = x.shape[3].value             
            weights1 = tf.get_variable(
                    shape = [1, size, in_filt, curr_filters[0]],
                    initializer = kernel_init,
                    name = "weight_"+str(r)+"_1")
            biases1 = tf.get_variable(
                    shape = [curr_filters[0]],
                    initializer = bias_init,
                    name="bias_"+str(r)+"_1")
            conv1 = tf.nn.conv2d(
                    x, weights1,
                    strides = [1, stride[0], stride[1], 1],
                    padding = padding,
                    name = "conv_"+str(r)+"_1")
            x = tf.nn.bias_add(conv1, biases1, name = "bias_add_"+str(r)+"_1")
            
            weights2 = tf.get_variable(
                    shape = [size, 1, curr_filters[0], curr_filters[1]],
                    initializer = kernel_init,
                    name = "weight_"+str(r)+"_2")
            biases2 = tf.get_variable(
                    shape = [curr_filters[1]],
                    initializer = bias_init,
                    name="bias_"+str(r)+"_2")
            conv2 = tf.nn.conv2d(
                    x, weights2,
                    strides = [1, stride[0], stride[1], 1],
                    padding = padding,
                    name = "conv_"+str(r)+"_2")
            x = tf.nn.bias_add(conv2, biases2, name = "bias_add_"+str(r)+"_2")
    return x


# group convolution with the same input and output depths
def group_conv2d_fixdepth(
        inputs, size, cardinality, group_width,
        stride = 1,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        name = "group_conv2d"):
    
    if cardinality == 1:
        return conv2d(
                inputs, size = size, n_filters = group_width,
                stride = stride,
                regularizer = regularizer,
                kernel_init = kernel_init,
                name = name)
        
            
    with tf.variable_scope(name):
        size_splits = [group_width]*cardinality
        groups = tf.split(inputs, size_splits, axis = 3, name = "split")
        conv_groups = []
        for i, group in enumerate(groups):
            conv = conv2d(
                    group, size = size, n_filters = group_width,
                    stride = stride,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv2d_"+str(i))
            conv_groups.append(conv)
            
        outputs = tf.concat(conv_groups, axis = 3, name = "concat")    
        
    return outputs


def max_pool2d(
        inputs,
        size = 2,
        stride = 2,
        padding = "SAME",
        name = "max_pool2d"):
    if not isinstance(size, (tuple, list)):
        size = [size, size]
    if not isinstance(stride, (tuple, list)):
        stride = [stride, stride]
    return tf.nn.max_pool(
            inputs,
            ksize = [1, size[0], size[1], 1],
            strides = [1,stride[0], stride[1], 1],
            padding = padding,
            name = name)


def avg_pool2d(
        inputs,
        size = 2,
        stride = 2,
        padding = "SAME",
        name = "avg_pool2d"):
    if not isinstance(size, (tuple, list)):
        size = [size, size]
    if not isinstance(stride, (tuple, list)):
        stride = [stride, stride]
    return tf.nn.avg_pool(
            inputs,
            ksize = [1, size[0], size[1], 1],
            strides = [1,stride[0], stride[1], 1],
            padding = padding,
            name = name)


def global_avg_pool2d(
        inputs,
        name = "global_avg_pool2d"):
    return tf.reduce_mean(inputs, axis = [1, 2], name = name)


def flatten(inputs, name = "flatten"):
    """Creates a flattening layer.
    Args:
        inputs: Input tensor.
        name: Name of the layer.
    Returns:
        1D Tensor.
    """    
    return tf.reshape(
            inputs,
            shape = [-1, np.prod(inputs.get_shape()[1:].as_list())],
            name = name)


def dense(
        inputs, n_units,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = tf.zeros_initializer(),
        name = "dense"):
    """Creates a fully-connected dense layer.
    Args:
        inputs: Input tensor.
        n_units: Number of units.
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        name: Name of the layer.
    Returns:
        1D Tensor.
    """    
    with tf.variable_scope(name):
        weights = tf.get_variable(
                shape = [inputs.shape[1].value, n_units],
                regularizer = regularizer,
                initializer = kernel_init,
                name = "weight")
        biases = tf.get_variable(
                shape = [n_units],
                initializer = bias_init,
                name = "bias")
        fc = tf.matmul(inputs, weights, name = "matmul")
        outputs = tf.nn.bias_add(fc, biases, name = "bias_add")
    return outputs
    

def dense_act(
        inputs, n_units, activation,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = tf.zeros_initializer(),
        name = "dense_act"):
    with tf.variable_scope(name):
        x = dense(
                inputs, n_units,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "dense")
        if activation is not None:
            x = activation(x, name = "activation")
    return x

dense_relu = partial(dense_act, activation = tf.nn.relu)
dense_sigmoid = partial(dense_act, activation = tf.nn.sigmoid)

def dense_bn(
        inputs, n_units,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = tf.zeros_initializer(),
        name = "dense_bn"):
    """Creates a fully-connected dense layer with batch normalization.
    Args:
        inputs: Input tensor.
        n_units: Number of units.
        is_training: A boolean or a TensorFlow boolean scalar tensor for
            indicating training or testing mode.
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        name: Name of the layer.        
    Returns:
        1D Tensor.
    """    
    with tf.variable_scope(name):
        x = dense(
                inputs, n_units,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "dense")
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm")
    return x


def dense_bn_act(
        inputs, n_units, activation,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = tf.zeros_initializer(),
        name = "dense_bn_act"):
    """Creates a fully-connected dense layer with batch normalization.
    Args:
        inputs: Input tensor.
        n_units: Number of units.
        is_training: A boolean or a TensorFlow boolean scalar tensor for
            indicating training or testing mode.
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        name: Name of the layer.        
    Returns:
        1D Tensor.
    """    
    with tf.variable_scope(name):
        x = dense(
                inputs, n_units,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "dense")
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm")
        if activation is not None:
            x = activation(x, name = "activation")        
    return x

dense_bn_relu = partial(dense_bn_act, activation=tf.nn.relu)


def dense_act_bn(
        inputs, n_units, activation,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(),
        bias_init = tf.zeros_initializer(),
        name = "dense_bn_act"):
    """Creates a fully-connected dense layer with batch normalization.
    Args:
        inputs: Input tensor.
        n_units: Number of units.
        is_training: A boolean or a TensorFlow boolean scalar tensor for
            indicating training or testing mode.
        kernel_init: Kernel initialization function.
        bias_init: Bias initialization function.
        name: Name of the layer.        
    Returns:
        1D Tensor.
    """    
    with tf.variable_scope(name):
        x = dense(
                inputs, n_units,
                regularizer = regularizer,
                kernel_init = kernel_init,
                bias_init = bias_init,
                name = "dense")
        if activation is not None:
            x = activation(x, name = "activation")        
        x = tf.layers.batch_normalization(
                x, training = is_training, name = "batch_norm")
    return x

dense_relu_bn = partial(dense_act_bn, activation=tf.nn.relu)


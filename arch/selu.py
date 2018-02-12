#!/usr/bin/env python2
# -*- coding: utf-8 -*-



import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
import numbers
from math import sqrt
import numpy as np


#https://medium.com/@damoncivin/self-normalising-neural-networks-snn-2a972c1d421

#source: https://github.com/bioinf-jku/SNNs

def selu(x, name="selu"):
    """ When using SELUs you have to keep the following in mind:
    # (1) scale inputs to zero mean and unit variance
    # (2) use SELUs
    # (3) initialize weights with stddev sqrt(1/n)
    # (4) use SELU dropout
    """
    with ops.name_scope(name) as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    
    
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))
    

def dense_selu_safe_initializer(x, seed = 42):
    input_dims = x.get_shape().as_list()
    f_in = input_dims[-1]
    return tf.truncated_normal_initializer(stddev=sqrt(1.0/f_in), dtype=tf.float32, seed = seed)


def conv_selu_safe_initializer(x, ksize, seed = 42):
    if isinstance(ksize, (list, tuple)):
        dims = ksize[0]*ksize[1]
    else:
        dims = ksize*ksize
    input_dims = x.get_shape().as_list()
    f_in = dims*input_dims[-1] # NHWC
    return tf.truncated_normal_initializer(stddev=sqrt(1.0/f_in), dtype=tf.float32, seed = seed)



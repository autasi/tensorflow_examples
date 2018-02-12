#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:13:37 2017

@author: autasi
"""

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math


def Kumar_normal(activation = "relu", mode = "FAN_AVG", seed = 42):
    """Kumar's weight initializer discussed in:
       Kumar, On Weight Initialization in Deep Neural Networks, 2017
       See https://arxiv.org/pdf/1704.08863.pdf
    Args:
        activation: Activation of the layer the weights are used for; "relu", 
            "sigmoid", and "tanh" are supported.
        mode: Which units to use, "FAN_IN", "FAN_OUT", "FAN_AVG"
        seed: Random seed number.
    Returns:
        A weight initializer.
    """    
    
    if activation is None:
        factor = 1.0
    elif activation == "relu":
        # stdev = sqrt(2.0/N) for normal, or sqrt(1.3*2.0/N) for truncated
        factor = 2.0
    elif activation == "sigmoid":
        # stdev = 3.6/sqrt(N) = sqrt(12.96/N) for normal, or sqrt(1.3*12.96/N) for truncated
        factor = 12.96
    elif activation == "tanh":
        # stdev = sqrt(1/N) for normal, or sqrt(1.3/N) for truncated
        factor = 1.0
    elif activation == "selu":
        # stdev = sqrt(1/N) for normal, or sqrt(1.3/N) for truncated
        factor = 1.0
    else:
        factor = 1.0
        
    return tf.contrib.layers.variance_scaling_initializer(
            factor = factor,
            mode = mode,
            uniform = False,
            seed = seed)


def He_normal(seed = 42):
    """He's truncated normal weight initializer discussed in:  
       He et al., Delving Deep into Rectifiers: Surpassing Human-level 
       Performance on Imagenet Classification, 2015
       See https://arxiv.org/pdf/1502.01852.pdf
    Args:
        seed: Random seed number.
    Returns:
        A weight initializer.
    """     
    return Kumar_normal(activation = "relu", mode = "FAN_AVG", seed = seed)



def variance_scaling_initializer_multi(
        factor = 2.0, mode = "FAN_IN", C = [1, 1], uniform = False,
        seed = None, dtype = dtypes.float32):
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point type.')
  if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
    raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

  def _initializer(shape, dtype=dtype, C=C, partition_info=None):
    """Initializer function."""
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)
    if mode == 'FAN_IN':
      # Count only number of input connections.
      n = C[0]*fan_in
    elif mode == 'FAN_OUT':
      # Count only number of output connections.
      n = C[1]*fan_out
    elif mode == 'FAN_AVG':
      # Average number of inputs and output connections.
      n = (C[0]*fan_in + C[1]*fan_out) / 2.0
    if uniform:
      # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
      limit = math.sqrt(3.0 * factor / n)
      return random_ops.random_uniform(shape, -limit, limit,
                                       dtype, seed=seed)
    else:
      # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
      trunc_stddev = math.sqrt(1.3 * factor / n)
      return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                         seed=seed)
  return _initializer



def Kumar_normal_multibranch(
        activation = "relu", mode = "FAN_AVG", C = [1, 1], seed = 42):
    """Kumar's weight initializer for multi-branch networks discussed in: 
       Kumar, On Weight Initialization in Deep Neural Networks, 2017
       See https://arxiv.org/pdf/1704.08863.pdf
       and
       Yang et al., Learning Feature Pyramids for Human Pose Estimation, 2017
       See https://arxiv.org/pdf/1708.01101.pdf
    Args:
        activation: Activation of the layer the weights are used for; "relu", 
            "sigmoid", and "tanh" are supported.
        mode: Which units to use, "FAN_IN", "FAN_OUT", "FAN_AVG"
        C: List of the numbers of input and the output branches.
        seed: Random seed number.
    Returns:
        A weight initializer.
    """    
    
    if activation is None:
        factor = 1.0
    elif activation == "relu":
        # stdev = sqrt(2.0/N) for normal, or sqrt(1.3*2.0/N) for truncated
        factor = 2.0
    elif activation == "sigmoid":
        # stdev = 3.6/sqrt(N) = sqrt(12.96/N) for normal, or sqrt(1.3*12.96/N) for truncated
        factor = 12.96
    elif activation == "tanh":
        # stdev = sqrt(1/N) for normal, or sqrt(1.3/N) for truncated
        factor = 1.0
    else:
        factor = 1.0
        
    return variance_scaling_initializer_multi(
            factor = factor,
            mode = mode,
            C = C,
            uniform = False,
            seed = seed)


def He_normal_multibranch(C = [1, 1], seed = 42):
    """He's truncated normal weight initializer for multi-branch networks
       discussed in:  
       He et al., Delving Deep into Rectifiers: Surpassing Human-level 
       Performance on Imagenet Classification, 2015
       See https://arxiv.org/pdf/1502.01852.pdf
    Args:
        seed: Random seed number.
    Returns:
        A weight initializer.
    """     
    return Kumar_normal_multibranch(
            activation = "relu", mode = "FAN_AVG", C = C, seed = seed)

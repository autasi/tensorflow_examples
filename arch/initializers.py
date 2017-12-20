#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:13:37 2017

@author: autasi
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math

def variance_scaling_initializer_multi(factor=2.0, mode='FAN_IN', C=[1,1], uniform=False,
                                 seed=None, dtype=dtypes.float32):
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point type.')
  if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
    raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

  # pylint: disable=unused-argument
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
  # pylint: enable=unused-argument

  return _initializer
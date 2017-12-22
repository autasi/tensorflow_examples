#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d, conv2d_bn, conv2d_factorized, conv2d_1d, dense
from arch.layers import max_pool, avg_pool
from arch.layers import Kumar_initializer

# grid module v1
def grid_module_v1(
        inputs,
        n_filters_1x1 = 96,
        n_reduce_3x3 = 64,
        n_filters_3x3 = 96,
        n_reduce_5x5 = 64,
        n_filters_5x5 = 96,
        n_filters_pool = 96,
        pool = max_pool,
        pool_size = 3,
        is_training = False,
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "inception_grid_v1"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d(
                    inputs, size=1, n_filters = n_filters_1x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d(
                    inputs, size=1, n_filters = n_reduce_3x3,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_1x1_3 = conv2d(
                    inputs, size=1, n_filters = n_reduce_5x5,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_3")
        
        x_3x3 = conv2d(
                    x_1x1_2, size=3, n_filters = n_filters_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3")
        
        x_5x5 = conv2d(
                    x_1x1_3, size=5, n_filters = n_filters_5x5,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_5x5")
        
        pool1 = pool(inputs, size=pool_size, stride=1, name="pool")
        x_1x1_4 = conv2d(
                    pool1, size=1, n_filters = n_filters_pool,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_4")
        
        inception = tf.nn.relu(tf.concat([x_1x1_1,x_3x3,x_5x5,x_1x1_4], axis=3))
        
    return inception


#https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
# https://arxiv.org/pdf/1512.00567v3.pdf
# add Figure 5 -> for the largest
# add Figure 6 -> for the 17x17
# add Figure 7 -> for the 8x8
# add Figure 8 -> add after 17x17
# add Figure 10 -> instead of pooling

# grid module v2
# 5x5 -> 3x3 + 3x3
def grid_module_v2(
        inputs,
        n_filters_1x1 = 96,
        n_reduce_3x3 = 64,
        n_filters_3x3 = 96,
        n_reduce_5x5 = 64,
        n_filters_5x5 = 96,
        n_filters_pool = 96,
        pool = max_pool,
        pool_size = 3,
        is_training = False,
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "inception_grid_v2"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d(
                    inputs, size=1, n_filters = n_filters_1x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d(
                    inputs, size=1, n_filters = n_reduce_3x3,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_1x1_3 = conv2d(
                    inputs, size=1, n_filters = n_reduce_5x5,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_3")
        
        x_3x3 = conv2d(
                    x_1x1_2, size=3, n_filters = n_filters_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3")
        
        x_3x3_1 = conv2d(
                    x_1x1_3, size=3, n_filters = n_filters_5x5,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3_1")

        x_3x3_2 = conv2d(
                    x_3x3_1, size=3, n_filters = n_filters_5x5,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_3x3_2")
        
        pool1 = pool(inputs, size=pool_size, stride=1, name="pool")
        x_1x1_4 = conv2d(
                    pool1, size=1, n_filters = n_filters_pool,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_4")
        
        x = tf.concat([x_1x1_1,x_3x3,x_3x3_2,x_1x1_4], axis=3)
        x = tf.layers.batch_normalization(x, training=is_training, name="batch_norm")
        inception = tf.nn.relu(x)
        
    return inception

# batch normalized grid module v2
def bn_grid_module_v2(
        inputs,
        n_filters_1x1 = 96,
        n_reduce_3x3 = 64,
        n_filters_3x3 = 96,
        n_reduce_5x5 = 64,
        n_filters_5x5 = 96,
        n_filters_pool = 96,
        pool = max_pool,
        pool_size = 3,
        is_training = False,
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "bn_inception_grid_v2"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d_bn(
                    inputs, size=1, n_filters = n_filters_1x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d_bn(
                    inputs, size=1, n_filters = n_reduce_3x3,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_1x1_3 = conv2d_bn(
                    inputs, size=1, n_filters = n_reduce_5x5,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_3")
        
        x_3x3 = conv2d_bn(
                    x_1x1_2, size=3, n_filters = n_filters_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3")
        
        x_3x3_1 = conv2d_bn(
                    x_1x1_3, size=3, n_filters = n_filters_5x5,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3_1")

        x_3x3_2 = conv2d_bn(
                    x_3x3_1, size=3, n_filters = n_filters_5x5,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_3x32_2")
        
        pool1 = pool(inputs, size=pool_size, stride=1, name="pool")
        x_1x1_4 = conv2d_bn(
                    pool1, size=1, n_filters = n_filters_pool,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_4")
        
        inception = tf.nn.relu(tf.concat([x_1x1_1,x_3x3,x_3x3_2,x_1x1_4], axis=3))
        
    return inception


# factorized grid module v2
# 7x7 is factorized to 1x7 + 7x1
def factorized_grid_module_v2(
        inputs,
        n = 7,
        n_filters_1x1 = 384,
        n_reduce_nxn = 192,
        n_filters_nxn = [[224, 256]],
        n_reduce_nxn2 = 192,
        n_filters_nxn2 = [[192, 224],[224, 256]],
        n_filters_pool = 128,
        pool = max_pool,
        pool_size = 3,
        is_training = False,
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "factorized_inception_grid"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d(
                    inputs, size=1, n_filters = n_filters_1x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d(
                    inputs, size=1, n_filters = n_reduce_nxn,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_1x1_3 = conv2d(
                    inputs, size=1, n_filters = n_reduce_nxn2,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_3")
        
        x_nxn = conv2d_factorized(
                    x_1x1_2, size=n, n_filters = n_filters_nxn,
                    n_repeat = 1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_nxn_fact")
        
        x_nxn2 = conv2d_factorized(
                    x_1x1_3, size=n, n_filters = n_filters_nxn2,
                    n_repeat = 2,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_nxn2_fact")
        
        pool1 = pool(inputs, size=pool_size, stride=1, name="pool")
        x_1x1_4 = conv2d(
                    pool1, size=1, n_filters = n_filters_pool,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_4")
        
        x = tf.concat([x_1x1_1,x_nxn,x_nxn2,x_1x1_4], axis=3)
        x = tf.layers.batch_normalization(x, training=is_training, name="batch_norm")
        inception = tf.nn.relu(x)
        
    return inception

# filterbank grid module v2
# 1x3 and 3x1 filters
def filterbank_grid_module_v2(
        inputs,
        n_filters_1x1 = 256,
        n_reduce_1_1x3_3x1 = 384,
        n_filters_1_1x3_3x1 = 256,
        n_reduce_3_1x3_3x1 = 384,
        n_filters_3_3x3 = 512,
        n_filters_3_1x3_3x1 = 256,
        n_filters_pool = 256,
        pool = max_pool,
        pool_size = 3,
        is_training = False,
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "inception_grid_filterbank"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d(
                    inputs, size=1, n_filters = n_filters_1x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d(
                    inputs, size=1, n_filters = n_reduce_1_1x3_3x1,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_1x1_3 = conv2d(
                    inputs, size=1, n_filters = n_reduce_3_1x3_3x1,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_3")
        
        x_1_13 = conv2d_1d(
                    x_1x1_2, size=3, dim=0, n_filters = n_filters_1_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_1_3x1")

        x_1_31 = conv2d_1d(
                    x_1x1_2, size=3, dim=1, n_filters = n_filters_1_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_1_1x3")
        
        x_3x3 = conv2d(
                    x_1x1_3, size=3, n_filters = n_filters_3_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3")
        
        x_3_13 = conv2d_1d(
                    x_3x3, size=3, dim=0, n_filters = n_filters_3_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3_3x1")

        x_3_31 = conv2d_1d(
                    x_3x3, size=3, dim=1, n_filters = n_filters_3_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3_1x3")
       
        pool1 = pool(inputs, size=pool_size, stride=1, name="pool")
        x_1x1_4 = conv2d(
                    pool1, size=1, n_filters = n_filters_pool,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_4")
        
        x = tf.concat([x_1x1_1,x_1_13,x_1_31,x_3_13,x_3_31,x_1x1_4], axis=3)
        x = tf.layers.batch_normalization(x, training=is_training, name="batch_norm")
        inception = tf.nn.relu(x)
        
    return inception

# reduction module v2
# concatenation of pooling and conv with stride 2
def reduction_module_v2(
        inputs,
        n_reduce_3x3 = 192,
        n_filters_3x3 = 384,
        n_reduce_3x3_2 = 192,
        n_filters_3x3_2_1 = 224,
        n_filters_3x3_2_2 = 256,
        pool = max_pool,
        pool_size = 3,
        is_training = False,        
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "reduction"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d(
                    inputs, size=1, n_filters = n_reduce_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d(
                    inputs, size=1, n_filters = n_reduce_3x3_2,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_3x3 = conv2d(
                    x_1x1_1, size=3, n_filters = n_filters_3x3,
                    stride = 2,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3")
        
        x_3x3_2_1 = conv2d(
                    x_1x1_2, size=3, n_filters = n_filters_3x3_2_1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3_2_1")

        x_3x3_2_2 = conv2d(
                    x_3x3_2_1, size=3, n_filters = n_filters_3x3_2_2,
                    stride = 2,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3_2_2")
        
        pool1 = pool(inputs, size=pool_size, stride=2, name="pool")

        x = tf.concat([x_3x3_2_2, x_3x3, pool1], axis=3)
        x = tf.layers.batch_normalization(x, training=is_training, name="batch_norm")
        inception = tf.nn.relu(x)
        
    return inception


# auxiliary_classifier_v3
# batch normalized pooling + conv + dense
def auxiliary_classifier_v3(
        inputs,
        pool = avg_pool,
        pool_size = 5,
        pool_stride = 3,
        n_filters_1x1 = 128,
        mode = "FAN_IN",
        n_units = 1024,
        is_training = False,
        name = "auxiliary_classifier"):
    
    with tf.variable_scope(name):    
        pool1 = pool(inputs, size=pool_size, stride=pool_stride, padding="VALID", name="pool")
        x_1x1 = conv2d(
                    pool1, size=1, n_filters = n_filters_1x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1")
        #flat = flatten(x_1x1, name="flatten")
        avg = tf.reduce_mean(x_1x1, [1,2])
        fc = dense(avg, n_units=n_units, activation=None, name="fc")
        outputs = tf.layers.batch_normalization(fc, training=is_training, name="batch_norm")
        outputs = tf.nn.relu(outputs)
        return outputs


# filterbank grid module v4
# 1x3 and 3x1 filters, 3x3 factorized
def filterbank_grid_module_v4(
        inputs,
        n_filters_1x1 = 256,
        n_reduce_1_1x3_3x1 = 384,
        n_filters_1_1x3_3x1 = 256,
        n_reduce_3_1x3_3x1 = 384,
        n_filters_3_3x3 = [[448, 512]],
        n_filters_3_1x3_3x1 = 256,
        n_filters_pool = 256,
        pool = max_pool,
        pool_size = 3,
        is_training = False,
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "inception_grid_filterbank"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d(
                    inputs, size=1, n_filters = n_filters_1x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d(
                    inputs, size=1, n_filters = n_reduce_1_1x3_3x1,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_1x1_3 = conv2d(
                    inputs, size=1, n_filters = n_reduce_3_1x3_3x1,
                    stride = 1,
                    activation = tf.nn.relu,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_3")
        
        x_1_13 = conv2d_1d(
                    x_1x1_2, size=3, dim=0, n_filters = n_filters_1_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_1_3x1")

        x_1_31 = conv2d_1d(
                    x_1x1_2, size=3, dim=1, n_filters = n_filters_1_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_1_1x3")
        
        x_3x3 = conv2d_factorized(
                    x_1x1_3, size=3, n_filters = n_filters_3_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3_fact")
        
        x_3_13 = conv2d_1d(
                    x_3x3, size=3, dim=0, n_filters = n_filters_3_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3_3x1")

        x_3_31 = conv2d_1d(
                    x_3x3, size=3, dim=1, n_filters = n_filters_3_1x3_3x1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3_1x3")
       
        pool1 = pool(inputs, size=pool_size, stride=1, name="pool")
        x_1x1_4 = conv2d(
                    pool1, size=1, n_filters = n_filters_pool,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_4")
        
        x = tf.concat([x_1x1_1,x_1_13,x_1_31,x_3_13,x_3_31,x_1x1_4], axis=3)
        x = tf.layers.batch_normalization(x, training=is_training, name="batch_norm")
        inception = tf.nn.relu(x)
        
    return inception


# reduction module v4
# concatenation of pooling and conv with stride 2
def reduction_module_v4(
        inputs,
        n_reduce_3x3 = 192,
        n_filters_3x3 = 192,
        n_reduce_7x7_3x3 = 256,
        n_filters_7x7 = [[256,320]],
        n_filters_7x7_3x3 = 320,
        pool = max_pool,
        pool_size = 3,
        is_training = False,        
        mode = "FAN_IN",
        mode_reduce = "FAN_AVG",
        name = "reduction"
        ):
    with tf.variable_scope(name):
        x_1x1_1 = conv2d(
                    inputs, size=1, n_filters = n_reduce_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_1")
        
        x_1x1_2 = conv2d(
                    inputs, size=1, n_filters = n_reduce_7x7_3x3,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode),
                    name = "conv2d_1x1_2")
        
        x_3x3 = conv2d(
                    x_1x1_1, size=3, n_filters = n_filters_3x3,
                    stride = 2,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_3x3")
        
        x_7x7 = conv2d_factorized(
                    x_1x1_2, size=7, n_filters = n_filters_7x7,
                    n_repeat = 1,
                    stride = 1,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_7x7_fact")

        x_7x7_3x3 = conv2d(
                    x_7x7, size=3, n_filters = n_filters_7x7_3x3,
                    stride = 2,
                    activation = None,
                    kernel_init = Kumar_initializer(mode=mode_reduce),
                    name = "conv2d_7x7_3x3")
        
        pool1 = pool(inputs, size=pool_size, stride=2, name="pool")

        x = tf.concat([x_7x7_3x3, x_3x3, pool1], axis=3)
        x = tf.layers.batch_normalization(x, training=is_training, name="batch_norm")
        inception = tf.nn.relu(x)
        
    return inception

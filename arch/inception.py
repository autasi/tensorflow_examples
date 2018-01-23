#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import tensorflow as tf
from arch.layers import conv2d_relu, conv2d_bn_relu, factorized_conv2d_bn_relu
from arch.layers import dense_relu, dense_bn_relu
from arch.layers import max_pool2d, avg_pool2d, flatten
from arch.initializers import He_normal


#Szegedy et al. Going deeper with convolutions
#https://arxiv.org/pdf/1409.4842v1.pdf
def grid(
        inputs,
        n_filters_1x1 = 64,
        n_reduce_3x3 = 96,
        n_filters_3x3 = 128,
        n_reduce_5x5 = 16,
        n_filters_5x5 = 32,
        n_filters_pool = 32,
        pool = max_pool2d,
        pool_size = 3,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_grid"
        ):
    with tf.variable_scope(name):
        # 1x1
        x_1x1 = conv2d_relu(
                    inputs, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x1")
        
        # 3x3
        reduce_3x3 = conv2d_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3")
        x_3x3 = conv2d_relu(
                    reduce_3x3, size = 3, n_filters = n_filters_3x3,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3")
        
        # 5x5
        reduce_5x5 = conv2d_relu(
                    inputs, size = 1, n_filters = n_reduce_5x5,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_5x5")
        x_5x5 = conv2d_relu(
                    reduce_5x5, size = 5, n_filters = n_filters_5x5,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_5x5")
        
        # pool projection
        pool = pool(inputs, size = pool_size, stride = 1, name = "pool")
        proj_pool = conv2d_relu(
                    pool, size = 1, n_filters = n_filters_pool,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_pool_proj")
        
        inception = tf.concat([x_1x1,x_3x3,x_5x5,proj_pool], axis = 3)
        
    return inception


# Szegedy et al., Rethinking the Inception Architecture for Computer Vision
# https://arxiv.org/pdf/1512.00567v3.pdf
# Figure 5
# Similar to grid_module, but the 5x5 is factorized to two 3x3 
def grid_fact2d(
        inputs,
        n_filters_1x1 = 64,
        n_reduce_3x3 = 96,
        n_filters_3x3 = 128,
        n_reduce_5x5 = 16,
        n_filters_5x5 = 32,
        n_filters_pool = 32,
        pool = max_pool2d,
        pool_size = 3,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_grid_2d_factorized"
        ):
    with tf.variable_scope(name):
        # 1x1
        x_1x1 = conv2d_relu(
                    inputs, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x1")
        
        # 3x3
        reduce_3x3 = conv2d_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3")
        x_3x3 = conv2d_relu(
                    reduce_3x3, size = 3, n_filters = n_filters_3x3,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3")
        
        # factorized 5x5
        reduce_5x5 = conv2d_relu(
                    inputs, size = 1, n_filters = n_reduce_5x5,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_5x5")
        fact_5x5_1 = conv2d_relu(
                    reduce_5x5, size = 3, n_filters = n_filters_5x5,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "fact_conv_5x5_1")
        fact_5x5_2 = conv2d_relu(
                    fact_5x5_1, size = 3, n_filters = n_filters_5x5,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "fact_conv_5x5_2")

        # pool projection
        pool = pool(inputs, size = pool_size, stride = 1, name = "pool")
        proj_pool = conv2d_relu(
                    pool, size = 1, n_filters = n_filters_pool,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_pool_proj")
        
        inception = tf.concat([x_1x1,x_3x3,fact_5x5_2,proj_pool], axis = 3)
        
    return inception

# Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by 
# Reducing Internal Covariate Shift
# http://proceedings.mlr.press/v37/ioffe15.pdf
# batch normalized version of the factorized grid module
def grid_fact2d_bn(
        inputs,
        n_filters_1x1 = 64,
        n_reduce_3x3 = 96,
        n_filters_3x3 = 128,
        n_reduce_5x5 = 16,
        n_filters_5x5 = 32,
        n_filters_pool = 32,
        pool = max_pool2d,
        pool_size = 3,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_grid_2d_factorized_batchnorm"
        ):
    with tf.variable_scope(name):
        # 1x1
        x_1x1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x1")
        
        # 3x3
        reduce_3x3 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3")
        x_3x3 = conv2d_bn_relu(
                    reduce_3x3, size = 3, n_filters = n_filters_3x3,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3")
        
        # factorized 5x5
        reduce_5x5 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_5x5,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_5x5")
        fact_5x5_1 = conv2d_bn_relu(
                    reduce_5x5, size = 3, n_filters = n_filters_5x5,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "fact_conv_5x5_1")
        fact_5x5_2 = conv2d_bn_relu(
                    fact_5x5_1, size = 3, n_filters = n_filters_5x5,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "fact_conv_5x5_2")

        # pool projection
        pool = pool(inputs, size = pool_size, stride = 1, name = "pool")
        proj_pool = conv2d_bn_relu(
                    pool, size = 1, n_filters = n_filters_pool,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_pool_proj")
        
        inception = tf.concat([x_1x1,x_3x3,fact_5x5_2,proj_pool], axis = 3)
        
    return inception


# Szegedy et al., Rethinking the Inception Architecture for Computer Vision
# https://arxiv.org/pdf/1512.00567v3.pdf
# Figure 6
# 7x7 is factorized to 1x7 and 7x1 convolutions
def grid_fact1d_bn(
        inputs,
        n_filters_1x1 = 384,
        n_reduce_7x7_1 = 192,
        n_filters_7x7_1 = [(224, 256)],
        n_reduce_7x7_2 = 192,
        n_filters_7x7_2 = [(192, 224),(224, 256)],
        n_filters_pool = 128,
        pool = max_pool2d,
        pool_size = 3,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_grid_1d_factorized_batchnorm"
        ):
    with tf.variable_scope(name):
        # 1x1
        x_1x1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x1")

        # 1x7 + 7x1
        reduce_7x7_1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_7x7_1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_7x7_1")
        
        x_7x7_1 = factorized_conv2d_bn_relu(
                    reduce_7x7_1, size = 7, n_repeat = 1,
                    n_filters = n_filters_7x7_1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_fact_7x7_1")
       
        # 1x7 + 7x1 + 1x7 + 7x1        
        reduce_7x7_2 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_7x7_2,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_7x7_2")


        x_7x7_2 = factorized_conv2d_bn_relu(
                    reduce_7x7_2, size = 7, n_repeat = 2,
                    n_filters = n_filters_7x7_2,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_fact_7x7_2")
        
        # pool projection
        pool = pool(inputs, size = pool_size, stride = 1, name = "pool")
        proj_pool = conv2d_bn_relu(
                    pool, size = 1, n_filters = n_filters_pool,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_pool_proj")
        
        inception = tf.concat([x_1x1,x_7x7_1,x_7x7_2,proj_pool], axis = 3)
        
    return inception


# Szegedy et al., Rethinking the Inception Architecture for Computer Vision
# https://arxiv.org/pdf/1512.00567v3.pdf
# Figure 7
# 1x3 and 3x1 filterbank
def expanded_filterbank_bn(
        inputs,
        n_filters_1x1 = 256,
        n_reduce_1x3_3x1 = 384,
        n_filters_1x3_3x1 = 256,
        n_reduce_3x3 = 384,
        n_filters_3x3 = 512,
        n_filters_3x3_1x3_3x1 = 256,
        n_filters_pool = 256,
        pool = max_pool2d,
        pool_size = 3,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_expanded_filterbank_batchnorm"
        ):
    with tf.variable_scope(name):
        # 1x1
        x_1x1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x1")
        
        # 1x3 + 3x1
        reduce_1x3_3x1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_1x3_3x1")
        x_1x3 = conv2d_bn_relu(
                    reduce_1x3_3x1, size = [1,3], n_filters = n_filters_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x3")
        x_3x1 = conv2d_bn_relu(
                    reduce_1x3_3x1, size = [3,1], n_filters = n_filters_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x1")
        
        # 3x3 + (1x3 + 3x1)
        reduce_3x3 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3")
        x_3x3 = conv2d_bn_relu(
                    reduce_3x3, size = 3, n_filters = n_filters_3x3,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3")
        x_3x3_1x3 = conv2d_relu(
                    x_3x3, size = [1,3], n_filters = n_filters_3x3_1x3_3x1,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_1x3")
        x_3x3_3x1 = conv2d_bn_relu(
                    x_3x3, size = [3,1], n_filters = n_filters_3x3_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_3x1")
       
        # pool projection
        pool = pool(inputs, size = pool_size, stride = 1, name = "pool")
        proj_pool = conv2d_bn_relu(
                    pool, size = 1, n_filters = n_filters_pool,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_pool_proj")
        
        inception = tf.concat([x_1x1,x_1x3,x_3x1,x_3x3_1x3,x_3x3_3x1,proj_pool], axis = 3)
        
    return inception


# Szegedy et al., Inception-v4, Inception-ResNet and the Impact of Residual 
# Connections on Learning
# https://arxiv.org/pdf/1602.07261.pdf
# Figure 6
# conv 3x3 factorized to 1x3 + 3x1
def expanded_filterbank_fact1d_bn(
        inputs,
        n_filters_1x1 = 256,
        n_reduce_1x3_3x1 = 384,
        n_filters_1x3_3x1 = 256,
        n_reduce_1x3 = 384,
        n_filters_1x3 = 448,
        n_filters_3x1 = 512,
        n_filters_3x3_1x3_3x1 = 256,
        n_filters_pool = 256,
        pool = avg_pool2d,
        pool_size = 3,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_expanded_filterbank_batchnorm"
        ):
    with tf.variable_scope(name):
        # 1x1
        x_1x1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x1")
        
        # 1x3 + 3x1
        reduce_1x3_3x1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_1x3_3x1")
        x_1x3 = conv2d_bn_relu(
                    reduce_1x3_3x1, size = [1,3], n_filters = n_filters_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x3_1")
        x_3x1 = conv2d_bn_relu(
                    reduce_1x3_3x1, size = [3,1], n_filters = n_filters_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x1_1")
        
        # 1x3 + 3x1 + (1x3 + 3x1)
        reduce_1x3 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_1x3,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_1x3")
        x_1x3_2 = conv2d_bn_relu(
                    reduce_1x3, size = [1,3], n_filters = n_filters_1x3,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x3_2")
        x_3x1_2 = conv2d_bn_relu(
                    x_1x3_2, size = [3,1], n_filters = n_filters_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x1_2")
        x_3x3_1x3 = conv2d_relu(
                    x_3x1_2, size = [1,3], n_filters = n_filters_3x3_1x3_3x1,
                    stride = 1,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_1x3")
        x_3x3_3x1 = conv2d_bn_relu(
                    x_3x1_2, size = [3,1], n_filters = n_filters_3x3_1x3_3x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_3x1")
       
        # pool projection
        pool = pool(inputs, size = pool_size, stride = 1, name = "pool")
        proj_pool = conv2d_bn_relu(
                    pool, size = 1, n_filters = n_filters_pool,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_pool_proj")
        
        inception = tf.concat([x_1x1,x_1x3,x_3x1,x_3x3_1x3,x_3x3_3x1,proj_pool], axis = 3)
        
    return inception


# Szegedy et al., Rethinking the Inception Architecture for Computer Vision
# https://arxiv.org/pdf/1512.00567v3.pdf
# Figure 10
# module to reduce grid size and expand filterbanks
def reduction_bn(
        inputs,
        n_reduce_3x3_1 = 192,
        n_filters_3x3_1 = 384,
        n_reduce_3x3_2 = 192,
        n_filters_3x3_2_1 = 224,
        n_filters_3x3_2_2 = 256,
        pool = max_pool2d,
        pool_size = 3,
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_reduction_batchnorm"
        ):
    with tf.variable_scope(name):
        # 3x3 stride 2
        reduce_3x3_1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3_1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3_1")
        x_3x3_1 = conv2d_bn_relu(
                    reduce_3x3_1, size = 3, n_filters = n_filters_3x3_1,
                    stride = 2,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_1")        
        
        # 3x3 stride 1 + 3x3 stride 2
        reduce_3x3_2 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3_2,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3_2")
        x_3x3_2_1 = conv2d_bn_relu(
                    reduce_3x3_2, size = 3, n_filters = n_filters_3x3_2_1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_2_1")
        x_3x3_2_2 = conv2d_bn_relu(
                    x_3x3_2_1, size = 3, n_filters = n_filters_3x3_2_2,
                    stride = 2,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_2_2")
        
        # pool
        pool = pool(inputs, size = pool_size, stride = 2, name = "pool")

        inception = tf.concat([x_3x3_1, x_3x3_2_2, pool], axis = 3)
        
    return inception


# Szegedy et al., Inception-v4, Inception-ResNet and the Impact of Residual 
# Connections on Learning
# https://arxiv.org/pdf/1602.07261.pdf
# Figure 7
# for 35 × 35 to 17 × 17
def reduction_bn_v4_1(
        inputs,
        n_filters_3x3_1 = 384, #n
        n_reduce_3x3_2 = 192, #k
        n_filters_3x3_2_1 = 224, #l
        n_filters_3x3_2_2 = 256, #m
        pool = max_pool2d,
        pool_size = 3,
        padding = "VALID",
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_v4_reduction_1"
        ):
    with tf.variable_scope(name):
        # 3x3 stride 2
        x_3x3_1 = conv2d_bn_relu(
                    inputs, size = 3, n_filters = n_filters_3x3_1,
                    stride = 2,
                    padding = padding,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_1")        
        
        # 3x3 stride 1 + 3x3 stride 2
        reduce_3x3_2 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3_2,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3_2")
        x_3x3_2_1 = conv2d_bn_relu(
                    reduce_3x3_2, size = 3, n_filters = n_filters_3x3_2_1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_2_1")
        x_3x3_2_2 = conv2d_bn_relu(
                    x_3x3_2_1, size = 3, n_filters = n_filters_3x3_2_2,
                    stride = 2,
                    padding = padding,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3_2_2")
        
        # pool
        pool = pool(
                inputs, size = pool_size, stride = 2,
                padding = padding, name = "pool")

        inception = tf.concat([x_3x3_1, x_3x3_2_2, pool], axis = 3)
        
    return inception


# Szegedy et al., Inception-v4, Inception-ResNet and the Impact of Residual 
# Connections on Learning
# https://arxiv.org/pdf/1602.07261.pdf
# Figure 8
# for 17 × 17 to 8 × 8
def reduction_bn_v4_2(
        inputs,
        n_reduce_3x3 = 192,
        n_filters_3x3 = 192,
        n_reduce_7x7 = 256,
        n_filters_1x7 = 256,
        n_filters_7x1 = 320,
        n_filters_7x7_3x3 = 320,
        pool = max_pool2d,
        pool_size = 3,
        padding = "VALID",
        is_training = False,
        regularizer = None,
        kernel_init = He_normal(seed = 42),
        kernel_init_reduce = He_normal(seed = 42),
        name = "inception_v4_reduction_2"
        ):
    with tf.variable_scope(name):
        # 3x3 stride 2
        reduce_3x3 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_3x3,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_3x3")
        x_3x3 = conv2d_bn_relu(
                    reduce_3x3, size = 3, n_filters = n_filters_3x3,
                    stride = 2,
                    padding = padding,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_3x3")        
        
        # 1x7 + 7x1 + 3x3 stride 2
        reduce_7x7 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_reduce_7x7,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init_reduce,
                    name = "conv_reduce_7x7")
        x_1x7 = conv2d_bn_relu(
                    reduce_7x7, size = [1,7], n_filters = n_filters_1x7,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_1x7")
        x_7x1 = conv2d_bn_relu(
                    x_1x7, size = [7,1], n_filters = n_filters_7x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_7x1")
        x_7x7_3x3 = conv2d_bn_relu(
                    x_7x1, size = 3, n_filters = n_filters_7x7_3x3,
                    stride = 2,
                    padding = padding,
                    is_training = is_training,
                    regularizer = regularizer,
                    kernel_init = kernel_init,
                    name = "conv_7x7_3x3")
        
        # pool
        pool = pool(
                inputs, size = pool_size, stride = 2,
                padding = padding, name = "pool")

        inception = tf.concat([x_3x3, x_7x7_3x3, pool], axis = 3)
        
    return inception




# Szegedy et al., Going deeper with convolutions
# https://arxiv.org/pdf/1409.4842.pdf
# pooling + conv 1x1 + dense + dropout
def auxiliary_classifier(
        inputs,
        pool = avg_pool2d,
        pool_size = 5,
        pool_stride = 3,
        n_filters_1x1 = 128,
        n_units = 1024,
        drop_rate = 0.7,
        seed = 42,
        is_training = False,
        regularizer_conv = None,
        regularizer_dense = None,
        kernel_init_conv = He_normal(seed = 42),
        kernel_init_dense = He_normal(seed = 42),
        name = "inception_auxiliary_classifier"):
    
    with tf.variable_scope(name): 
        # pool
        pool = pool(
                inputs, size = pool_size, stride = pool_stride, 
                padding = "VALID", name = "pool")
        
        # 1x1
        x_1x1 = conv2d_relu(
                    pool, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    regularizer = regularizer_conv,
                    kernel_init = kernel_init_conv,
                    name = "conv_1x1")

        # dense
        flat = flatten(x_1x1, name = "flatten")
        dense = dense_relu(
                flat, n_units = n_units,
                regularizer = regularizer_dense,
                kernel_init = kernel_init_dense,
                name = "dense")
        
        # dropout
        if drop_rate > 0.0:
            dense = tf.layers.dropout(
                    dense, rate = drop_rate,
                    training = is_training,
                    seed = seed, name = "dropout")
        
        return dense


# Szegedy et al., Rethinking the Inception Architecture for Computer Vision
# Figure 8
# batch normalized pooling + conv 1x1 + dense
def auxiliary_classifier_bn(
        inputs,
        pool = avg_pool2d,
        pool_size = 5,
        pool_stride = 3,
        n_filters_1x1 = 128,
        n_units = 1024,
        is_training = False,
        regularizer_conv = None,
        regularizer_dense = None,
        kernel_init_conv = He_normal(seed = 42),
        kernel_init_dense = He_normal(seed = 42),
        name = "inception_auxiliary_classifier_batchnorm"):
    
    with tf.variable_scope(name): 
        # pool
        pool = pool(
                inputs, size = pool_size, stride = pool_stride, 
                padding = "VALID", name = "pool")
        
        # 1x1
        x_1x1 = conv2d_bn_relu(
                    inputs, size = 1, n_filters = n_filters_1x1,
                    stride = 1,
                    is_training = is_training,
                    regularizer = regularizer_conv,
                    kernel_init = kernel_init_conv,
                    name = "conv_1x1")

        # dense
        flat = flatten(x_1x1, name = "flatten")
        dense = dense_bn_relu(
                flat, n_units = n_units,
                is_training = is_training,
                regularizer = regularizer_dense,
                kernel_init = kernel_init_dense,
                name = "dense")
                
        return dense


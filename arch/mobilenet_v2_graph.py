# -*- coding: utf-8 -*-
import tensorflow as tf
from arch.layers import conv2d_bn_relu, dense, global_avg_pool2d
from arch.initializers import He_normal, Kumar_normal
import mobilenet_v2

#Sandler et al., Inverted Residuals and Linear Bottlenecks: Mobile Networks for
#Classification, Detection and Segmentation
#https://arxiv.org/abs/1801.04381v2
def cifar10_mobilenet_v2(x, expand_ratio = 6, n_filters = 32, n_repeat = 3, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv1 = conv2d_bn_relu(
            x, size = 3, n_filters = n_filters,
            kernel_init = He_normal(seed = seed+1),
            is_training = training,
            name = "initial_conv")
    layers.append(("initial_conv", conv1))
    
    # 32x32x32 -> 32x32x32
    invres1 = mobilenet_v2.inverted_residual_block(
            conv1, n_repeat = n_repeat,
            n_filters = n_filters, expand_ratio = expand_ratio,
            stride = 1,
            kernel_init = He_normal(seed = seed+2),
            is_training = training,
            name = "inverted_residual_block_1")
    layers.append(("inverted_residual_block_1", invres1))
    
    # 32x32x32 -> 16x16x64
    invres2 = mobilenet_v2.inverted_residual_block(
            invres1, n_repeat = n_repeat,
            n_filters = n_filters*2, expand_ratio = expand_ratio,
            stride = 2,
            kernel_init = He_normal(seed = seed+3),
            is_training = training,
            name = "inverted_residual_block_2")
    layers.append(("inverted_residual_block_2", invres2))

    #16x16x64 -> 8x8x96
    invres3 = mobilenet_v2.inverted_residual_block(
            invres2, n_repeat = n_repeat,
            n_filters = n_filters*3, expand_ratio = expand_ratio,
            stride = 2,
            kernel_init = He_normal(seed = seed+4),
            is_training = training,
            name = "inverted_residual_block_3")
    layers.append(("inverted_residual_block_3", invres3))
    
    #8x8x96 -> 8x8x384    
    conv2 = conv2d_bn_relu(
            invres3, size = 1, n_filters = n_filters*3*4,
            kernel_init = He_normal(seed = seed+5),
            is_training = training,
            name = "final_conv")
    layers.append(("final_conv", conv2))
    
    pool = global_avg_pool2d(conv2)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables
    
    
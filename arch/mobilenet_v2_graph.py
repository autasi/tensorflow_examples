# -*- coding: utf-8 -*-
import tensorflow as tf
from arch.layers import conv2d_bn_relu, dense, global_avg_pool2d
from arch.initializers import He_normal, Kumar_normal
from arch import mobilenet_v2

#Sandler et al., Inverted Residuals and Linear Bottlenecks: Mobile Networks for
#Classification, Detection and Segmentation
#https://arxiv.org/abs/1801.04381v2
#def cifar10_mobilenet_v2(x, expand_ratio = 6, n_filters = 32, n_repeat = 3, seed = 42):
#    layers = []
#    variables = []
#
#    training = tf.placeholder(tf.bool, name="training")
#    variables.append(("training", training))
#
#    conv1 = conv2d_bn_relu(
#            x, size = 3, n_filters = n_filters,
#            kernel_init = He_normal(seed = seed+1),
#            is_training = training,
#            name = "initial_conv")
#    layers.append(("initial_conv", conv1))
#    
#    # 32x32x32 -> 32x32x32
#    invres1 = mobilenet_v2.inverted_residual_block(
#            conv1, n_repeat = n_repeat,
#            n_filters = n_filters, expand_ratio = expand_ratio,
#            stride = 1,
#            kernel_init = He_normal(seed = seed+2),
#            is_training = training,
#            name = "inverted_residual_block_1")
#    layers.append(("inverted_residual_block_1", invres1))
#    
#    # 32x32x32 -> 16x16x64
#    invres2 = mobilenet_v2.inverted_residual_block(
#            invres1, n_repeat = n_repeat,
#            n_filters = n_filters*2, expand_ratio = expand_ratio,
#            stride = 2,
#            kernel_init = He_normal(seed = seed+3),
#            is_training = training,
#            name = "inverted_residual_block_2")
#    layers.append(("inverted_residual_block_2", invres2))
#
#    #16x16x64 -> 8x8x96
#    invres3 = mobilenet_v2.inverted_residual_block(
#            invres2, n_repeat = n_repeat,
#            n_filters = n_filters*3, expand_ratio = expand_ratio,
#            stride = 2,
#            kernel_init = He_normal(seed = seed+4),
#            is_training = training,
#            name = "inverted_residual_block_3")
#    layers.append(("inverted_residual_block_3", invres3))
#    
#    #8x8x96 -> 8x8x384    
#    conv2 = conv2d_bn_relu(
#            invres3, size = 1, n_filters = n_filters*3*4,
#            kernel_init = He_normal(seed = seed+5),
#            is_training = training,
#            name = "final_conv")
#    layers.append(("final_conv", conv2))
#    
#    pool = global_avg_pool2d(conv2)
#    layers.append(("pool", pool))
#    
#    dense1 = dense(
#            pool, n_units = 10,
#            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+5),
#            name = "dense_1")
#    layers.append(("logit", dense1))
#    
#    prob = tf.nn.softmax(dense1, name = "prob")
#    layers.append(("prob", prob))
#    
#    return layers, variables
#    
#    


def cifar10_mobilenet_v2(x, expand_ratio = 6, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv1 = conv2d_bn_relu(
            x, size = 3, n_filters = 32,
            kernel_init = He_normal(seed = seed+1),
            is_training = training,
            name = "initial_conv")
    layers.append(("initial_conv", conv1))
    
    # 32x32x32 -> 32x32x16
    invres1 = mobilenet_v2.inverted_residual_block(
            conv1, n_repeat = 1, n_filters = 16, stride = 1, expand_ratio = 1,
            kernel_init = He_normal(seed = seed+2),
            is_training = training,
            name = "inverted_residual_block_1")
    layers.append(("inverted_residual_block_1", invres1))
    
    
    # 32x32x16 -> 32x32x24
    invres2 = mobilenet_v2.inverted_residual_block(
            invres1, n_repeat = 2, n_filters = 24, stride = 1,
            expand_ratio = expand_ratio,
            kernel_init = He_normal(seed = seed+3),
            is_training = training,
            name = "inverted_residual_block_2")
    layers.append(("inverted_residual_block_2", invres2))


    #32x32x24 -> 16x16x32
    invres3 = mobilenet_v2.inverted_residual_block(
            invres2, n_repeat = 3, n_filters = 32, stride = 2,
            expand_ratio = expand_ratio,
            kernel_init = He_normal(seed = seed+4),
            is_training = training,
            name = "inverted_residual_block_3")
    layers.append(("inverted_residual_block_3", invres3))
    
    #16x16x32 -> 8x8x64
    invres4 = mobilenet_v2.inverted_residual_block(
            invres3, n_repeat = 4, n_filters = 64, stride = 2,
            expand_ratio = expand_ratio,
            kernel_init = He_normal(seed = seed+5),
            is_training = training,
            name = "inverted_residual_block_4")
    layers.append(("inverted_residual_block_4", invres4))
    
    #8x8x64 -> 8x8x96
    invres5 = mobilenet_v2.inverted_residual_block(
            invres4, n_repeat = 3, n_filters = 96, stride = 1,
            expand_ratio = expand_ratio,
            kernel_init = He_normal(seed = seed+6),
            is_training = training,
            name = "inverted_residual_block_5")
    layers.append(("inverted_residual_block_5", invres5))

    #8x8x96 -> 4x4x160
    invres6 = mobilenet_v2.inverted_residual_block(
            invres5, n_repeat = 3, n_filters = 160, stride = 2,
            expand_ratio = expand_ratio,
            kernel_init = He_normal(seed = seed+7),
            is_training = training,
            name = "inverted_residual_block_6")
    layers.append(("inverted_residual_block_6", invres6))

    #4x4x160 -> 4x4x320
    invres7 = mobilenet_v2.inverted_residual_block(
            invres6, n_repeat = 1, n_filters = 320, stride = 1,
            expand_ratio = expand_ratio,
            kernel_init = He_normal(seed = seed+8),
            is_training = training,
            name = "inverted_residual_block_7")
    layers.append(("inverted_residual_block_7", invres7))

    conv2 = conv2d_bn_relu(
            invres7, size = 1, n_filters = 1280,
            kernel_init = He_normal(seed = seed+9),
            is_training = training,
            name = "final_conv")
    layers.append(("final_conv", conv2))
    
    pool = global_avg_pool2d(conv2)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+10),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables
    

def cifar10_mobilenet_v2_wd(x, expand_ratio = 6, weight_decay = 0.0001, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv1 = conv2d_bn_relu(
            x, size = 3, n_filters = 32,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            is_training = training,
            name = "initial_conv")
    layers.append(("initial_conv", conv1))
    
    # 32x32x32 -> 32x32x16
    invres1 = mobilenet_v2.inverted_residual_block(
            conv1, n_repeat = 1, n_filters = 16, stride = 1, expand_ratio = 1,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2),
            is_training = training,
            name = "inverted_residual_block_1")
    layers.append(("inverted_residual_block_1", invres1))
    
    
    # 32x32x16 -> 32x32x24
    invres2 = mobilenet_v2.inverted_residual_block(
            invres1, n_repeat = 2, n_filters = 24, stride = 1,
            expand_ratio = expand_ratio,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+3),
            is_training = training,
            name = "inverted_residual_block_2")
    layers.append(("inverted_residual_block_2", invres2))


    #32x32x24 -> 16x16x32
    invres3 = mobilenet_v2.inverted_residual_block(
            invres2, n_repeat = 3, n_filters = 32, stride = 2,
            expand_ratio = expand_ratio,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4),
            is_training = training,
            name = "inverted_residual_block_3")
    layers.append(("inverted_residual_block_3", invres3))
    
    #16x16x32 -> 8x8x64
    invres4 = mobilenet_v2.inverted_residual_block(
            invres3, n_repeat = 4, n_filters = 64, stride = 2,
            expand_ratio = expand_ratio,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+5),
            is_training = training,
            name = "inverted_residual_block_4")
    layers.append(("inverted_residual_block_4", invres4))
    
    #8x8x64 -> 8x8x96
    invres5 = mobilenet_v2.inverted_residual_block(
            invres4, n_repeat = 3, n_filters = 96, stride = 1,
            expand_ratio = expand_ratio,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+6),
            is_training = training,
            name = "inverted_residual_block_5")
    layers.append(("inverted_residual_block_5", invres5))

    #8x8x96 -> 4x4x160
    invres6 = mobilenet_v2.inverted_residual_block(
            invres5, n_repeat = 3, n_filters = 160, stride = 2,
            expand_ratio = expand_ratio,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+7),
            is_training = training,
            name = "inverted_residual_block_6")
    layers.append(("inverted_residual_block_6", invres6))

    #4x4x160 -> 4x4x320
    invres7 = mobilenet_v2.inverted_residual_block(
            invres6, n_repeat = 1, n_filters = 320, stride = 1,
            expand_ratio = expand_ratio,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+8),
            is_training = training,
            name = "inverted_residual_block_7")
    layers.append(("inverted_residual_block_7", invres7))

    conv2 = conv2d_bn_relu(
            invres7, size = 1, n_filters = 1280,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+9),
            is_training = training,
            name = "final_conv")
    layers.append(("final_conv", conv2))
    
    pool = global_avg_pool2d(conv2)
    layers.append(("pool", pool))
    
    dense1 = dense(
            pool, n_units = 10,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+10),
            name = "dense_1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables
    

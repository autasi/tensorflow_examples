# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d_bn_relu, dense, global_avg_pool2d
from arch.initializers import He_normal, Kumar_normal
import mobilenet

#MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
#https://arxiv.org/abs/1704.04861
#https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
#https://github.com/Zehaos/MobileNet
def cifar10_mobilenet(x, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 32,
            kernel_init = He_normal(seed = seed+1),
            is_training = training,
            name = "initial_conv")
    layers.append(("initial_conv", conv))
    
    mblock1 = mobilenet.mobilenet_block(
            conv, n_filters = 64, stride = 1,
            kernel_init = He_normal(seed = seed+2),
            is_training = training,
            name = "mobilenet_block_1"
            )            
    layers.append(("mobilenet_block_1", mblock1))

    # 16x16
    mblock2 = mobilenet.mobilenet_block(
            mblock1, n_filters = 128, stride = 2,
            kernel_init = He_normal(seed = seed+3),
            is_training = training,
            name = "mobilenet_block_2"
            )            
    layers.append(("mobilenet_block_2", mblock2))

    mblock3 = mobilenet.mobilenet_block(
            mblock2, n_filters = 128, stride = 1,
            kernel_init = He_normal(seed = seed+4),
            is_training = training,
            name = "mobilenet_block_3"
            )            
    layers.append(("mobilenet_block_3", mblock3))

    # 8x8
    mblock4 = mobilenet.mobilenet_block(
            mblock3, n_filters = 256, stride = 2,
            kernel_init = He_normal(seed = seed+5),
            is_training = training,
            name = "mobilenet_block_4"
            )            
    layers.append(("mobilenet_block_4", mblock4))

    mblock5 = mobilenet.mobilenet_block(
            mblock4, n_filters = 256, stride = 1,
            kernel_init = He_normal(seed = seed+6),
            is_training = training,
            name = "mobilenet_block_5"
            )            
    layers.append(("mobilenet_block_5", mblock5))

    # 4x4
    mblock6 = mobilenet.mobilenet_block(
            mblock5, n_filters = 512, stride = 2,
            kernel_init = He_normal(seed = seed+7),
            is_training = training,
            name = "mobilenet_block_6"
            )            
    layers.append(("mobilenet_block_6", mblock6))

    mblock7 = mobilenet.mobilenet_block(
            mblock6, n_filters = 512, stride = 1,
            kernel_init = He_normal(seed = seed+8),
            is_training = training,
            name = "mobilenet_block_7"
            )            
    layers.append(("mobilenet_block_7", mblock7))

    mblock8 = mobilenet.mobilenet_block(
            mblock7, n_filters = 512, stride = 1,
            kernel_init = He_normal(seed = seed+9),
            is_training = training,
            name = "mobilenet_block_8"
            )            
    layers.append(("mobilenet_block_8", mblock8))
    
    mblock9 = mobilenet.mobilenet_block(
            mblock8, n_filters = 512, stride = 1,
            kernel_init = He_normal(seed = seed+10),
            is_training = training,
            name = "mobilenet_block_9"
            )            
    layers.append(("mobilenet_block_9", mblock9))
    
    mblock10 = mobilenet.mobilenet_block(
            mblock9, n_filters = 512, stride = 1,
            kernel_init = He_normal(seed = seed+11),
            is_training = training,
            name = "mobilenet_block_10"
            )            
    layers.append(("mobilenet_block_10", mblock10))
    
    mblock11 = mobilenet.mobilenet_block(
            mblock10, n_filters = 512, stride = 1,
            kernel_init = He_normal(seed = seed+12),
            is_training = training,
            name = "mobilenet_block_11"
            )            
    layers.append(("mobilenet_block_11", mblock11))

    # 2x2
    mblock12 = mobilenet.mobilenet_block(
            mblock11, n_filters = 1024, stride = 2,
            kernel_init = He_normal(seed = seed+13),
            is_training = training,
            name = "mobilenet_block_12"
            )            
    layers.append(("mobilenet_block_12", mblock12))

    mblock13 = mobilenet.mobilenet_block(
            mblock12, n_filters = 1024, stride = 1,
            kernel_init = He_normal(seed = seed+14),
            is_training = training,
            name = "mobilenet_block_13"
            )            
    layers.append(("mobilenet_block_13", mblock13))

    pool = global_avg_pool2d(mblock13)
    layers.append(("pool", pool))
    
    dense1 = dense(
                pool, n_units = 10,
                kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+15),
                name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))

    
    return layers, variables

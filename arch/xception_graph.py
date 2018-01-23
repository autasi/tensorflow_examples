# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d_bn_relu, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal
import xception

# F. Chollet, Xception: Deep Learning with Depthwise Separable Convolutions
# https://arxiv.org/pdf/1610.02357.pdf
# using 3 middle modules instead of 8
def cifar10_xception(x, drop_rate = 0.5, seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))

    # 32x32
    conv = conv2d_bn_relu(
            x, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name="initial_conv")
    layers.append(("initial_conv", conv))

    entry = xception.entry_module(
            conv,
            n_filters = [128, 256, 728],
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "entry")
    layers.append(("entry", entry))

    mid = xception.middle_module(
            entry,
            n_filters = 728,
            n_repeat = 3,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "middle")
    layers.append(("middle", mid))

    exits = xception.exit_module(
            mid,
            n_filters_1 = [728, 1024],
            n_filters_2 = [1536, 2048],
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "exit")
    layers.append(("exit", exits))


    pool1 = global_avg_pool2d(exits, name = "pool1")
    layers.append(("pool1", pool1))
             
    dropout1 = tf.layers.dropout(
            pool1, rate = drop_rate, training = training, 
            seed = seed+5, name = "dropout")
    layers.append(("dropout1", dropout1))

    dense1 = dense(
            dropout1, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+6),
            name = "dense1")
    layers.append(("logit", dense1))
    
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables

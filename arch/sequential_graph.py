#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d_relu, conv2d_relu_bn, dense_relu, dense_relu_bn
from arch.layers import dense, max_pool2d, global_avg_pool2d, flatten
from arch.initializers import He_normal, Kumar_normal


def cifar10_sequential_cn3d(
        x,
        drop_rate_1 = 0.2,
        drop_rate_2 = 0.3,
        drop_rate_3 = 0.4,
        seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name = "training")
    variables.append(("training", training))

    # conv5x5 relu + lrn + pool
    conv1 = conv2d_relu(
            x, size = 5, n_filters = 32,
            kernel_init = tf.truncated_normal_initializer(stddev = 5e-2, seed = seed+1),
            name = "conv_1")
    layers.append(("conv_1", conv1))
    norm1 = tf.nn.lrn(
            conv1,
            depth_radius = 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75,
            name = "norm_1")
    layers.append(("norm_1", norm1))            
    pool1 = max_pool2d(norm1, size = 3, stride = 2, name = "pool_1")
    layers.append(("pool_1", pool1))
    dropout1 = tf.layers.dropout(
            pool1, rate = drop_rate_1, training = training,
            seed = seed+1, name = "dropout_1")
    layers.append(("dropout_1", dropout1))    

    # conv5x5 relu + lrn + pool
    conv2 = conv2d_relu(
            dropout1, size = 5, n_filters = 64,
            kernel_init = tf.truncated_normal_initializer(stddev = 5e-2, seed = seed+2),
            name = "conv_2")
    layers.append(("conv_2", conv2))            
    norm2 = tf.nn.lrn(
            conv2,
            depth_radius = 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75,
            name = "norm_2")
    layers.append(("norm_2", norm2))
    pool2 = max_pool2d(norm2, size = 3, stride = 2, name="pool_2")
    layers.append(("pool_2", pool2))
    dropout2 = tf.layers.dropout(
            pool2, rate = drop_rate_2, training = training,
            seed = seed+2, name = "dropout_2")
    layers.append(("dropout_2", dropout2))    

    # conv3x3 relu + lrn + pool
    conv3 = conv2d_relu(
            dropout2, size = 3, n_filters = 128,
            kernel_init = tf.truncated_normal_initializer(stddev = 5e-2, seed = seed+3),
            name = "conv_3")
    layers.append(("conv_3", conv3))
    norm3 = tf.nn.lrn(
            conv3,
            depth_radius = 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75,
            name = "norm_3")
    layers.append(("norm_3", norm3))
    pool3 = max_pool2d(norm3, size = 2, stride = 2, name="pool_3")
    layers.append(("pool_3", pool3))
    dropout3 = tf.layers.dropout(
            pool3, rate = drop_rate_3, training = training,
            seed = seed+3, name = "dropout_3")
    layers.append(("dropout_3", dropout3))        

    flat = flatten(dropout3, name = "flatten")
    layers.append(("flatten", flat))

    # dense softmax
    dense1 = dense(
            flat, n_units = 10,
            kernel_init = tf.truncated_normal_initializer(stddev = 1/192.0, seed = seed+4),
            name = "dense_1")
    layers.append(("logit", dense1))
    prob = tf.nn.softmax(dense1, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables


def cifar10_sequential_cbn3d(
        x,
        drop_rate_1 = 0.2,
        drop_rate_2 = 0.3,
        drop_rate_3 = 0.4,
        seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name = "training")
    variables.append(("training", training))

    # conv5x5 relu batch-norm + pool
    conv1 = conv2d_relu_bn(
            x, size = 5, n_filters = 32,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name = "conv_1")
    layers.append(("conv_1", conv1))                
    pool1 = max_pool2d(conv1, size = 3, stride = 2, name = "pool_1")
    layers.append(("pool_1", pool1))
    dropout1 = tf.layers.dropout(
            pool1, rate = drop_rate_1, training = training,
            seed = seed+1, name = "dropout_1")
    layers.append(("dropout_1", dropout1))    

    # conv5x5 relu batch-norm + pool
    conv2 = conv2d_relu_bn(
            dropout1, size = 5, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "conv_2")
    layers.append(("conv_2", conv2))            
    pool2 = max_pool2d(conv2, size = 3, stride = 2, name = "pool_2")
    layers.append(("pool_2", pool2))
    dropout2 = tf.layers.dropout(
            pool2, rate = drop_rate_2, training = training,
            seed = seed+2, name = "dropout_2")
    layers.append(("dropout_2", dropout2))    

    # conv3x3 relu batch-norm + pool
    conv3 = conv2d_relu_bn(
            dropout2, size = 3, n_filters = 128,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "conv_3")
    layers.append(("conv_3", conv3))
    pool3 = max_pool2d(conv3, size = 2, stride = 2, name="pool_3")
    layers.append(("pool_3", pool3))
    dropout3 = tf.layers.dropout(
            pool3, rate = drop_rate_3, training = training,
            seed = seed+3, name = "dropout_3")
    layers.append(("dropout_3", dropout3))    
    
    flat = flatten(dropout3, name="flatten")
    layers.append(("flatten", flat))

    # dense softmax    
    dense1 = dense(
            flat, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+4),
            name = "dense_1")
    layers.append(("logit", dense1))    
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables


def cifar10_sequential_cbn6d(
        x,
        drop_rate_1 = 0.2,
        drop_rate_2 = 0.3,
        drop_rate_3 = 0.4,
        seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name = "training")
    variables.append(("training", training))

    # 2x conv3x3 relu batch-norm + pool
    conv1 = conv2d_relu_bn(
            x, size = 3, n_filters = 32,
            is_training = training,
            kernel_init = He_normal(seed = seed+1),
            name = "conv_1")
    layers.append(("conv_1", conv1))    
    conv2 = conv2d_relu_bn(
            conv1, size = 3, n_filters = 32,
            is_training = training,
            kernel_init = He_normal(seed = seed+2),
            name = "conv_2")
    layers.append(("conv_2", conv2))    
    pool1 = max_pool2d(conv2, size = 2, stride = 2, name = "pool_1")
    layers.append(("pool_1", pool1))
    dropout1 = tf.layers.dropout(
            pool1, rate = drop_rate_1, training = training,
            seed = seed+2, name = "dropout_1")
    layers.append(("dropout_1", dropout1))
    
    # 2x conv3x3 relu batch-norm + pool
    conv3 = conv2d_relu_bn(
            dropout1, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+3),
            name = "conv_3")
    layers.append(("conv_3", conv3))
    conv4 = conv2d_relu_bn(
            conv3, size = 3, n_filters = 64,
            is_training = training,
            kernel_init = He_normal(seed = seed+4),
            name = "conv_4")
    layers.append(("conv_4", conv4))
    pool2 = max_pool2d(conv4, size = 2, stride = 2, name = "pool_2")
    layers.append(("pool_2", pool2))
    dropout2 = tf.layers.dropout(
            pool2, rate = drop_rate_2, training = training,
            seed = seed+4, name = "dropout_2")
    layers.append(("dropout_2", dropout2))

    # 2x conv3x3 relu batch-norm + pool
    conv5 = conv2d_relu_bn(
            dropout2, size = 3, n_filters = 128,
            is_training = training,
            kernel_init = He_normal(seed = seed+5),
            name = "conv_5")
    layers.append(("conv_5", conv5))
    conv6 = conv2d_relu_bn(
            conv5, size = 3, n_filters = 128,
            is_training = training,
            kernel_init = He_normal(seed = seed+6),
            name = "conv_6")
    layers.append(("conv_6", conv6))
    pool3 = max_pool2d(conv6, size = 2, stride = 2, name="pool_3")
    layers.append(("pool_3", pool3))
    dropout3 = tf.layers.dropout(
            pool3, rate = drop_rate_3, training = training,
            seed = seed+6, name = "dropout_3")
    layers.append(("dropout_3", dropout3))

    flat = flatten(dropout3, name="flatten")
    layers.append(("flatten", flat))
    
    # dense softmax
    dense1 = dense(
            flat, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+7),
            name = "dense_1")
    layers.append(("logit", dense1))
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables


# https://www.kaggle.com/c/cifar-10/discussion/40237
def cifar10_sequential_cbn6d_wd(
        x,
        drop_rate_1 = 0.2,
        drop_rate_2 = 0.3,
        drop_rate_3 = 0.4,
        weight_decay = 0.0001,
        seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name = "training")
    variables.append(("training", training))

    # 2x conv3x3 relu batch-norm + pool
    conv1 = conv2d_relu_bn(
            x, size = 3, n_filters = 32,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name = "conv_1")
    layers.append(("conv_1", conv1))    
    conv2 = conv2d_relu_bn(
            conv1, size = 3, n_filters = 32,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2),
            name = "conv_2")
    layers.append(("conv_2", conv2))    
    pool1 = max_pool2d(conv2, size = 2, stride = 2, name = "pool_1")
    layers.append(("pool_1", pool1))
    dropout1 = tf.layers.dropout(
            pool1, rate = drop_rate_1, training = training,
            seed = seed+2, name = "dropout_1")
    layers.append(("dropout_1", dropout1))
    
    # 2x conv3x3 relu batch-norm + pool
    conv3 = conv2d_relu_bn(
            dropout1, size = 3, n_filters = 64,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+3),
            name = "conv_3")
    layers.append(("conv_3", conv3))
    conv4 = conv2d_relu_bn(
            conv3, size = 3, n_filters = 64,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4),
            name = "conv_4")
    layers.append(("conv_4", conv4))
    pool2 = max_pool2d(conv4, size = 2, stride = 2, name = "pool_2")
    layers.append(("pool_2", pool2))
    dropout2 = tf.layers.dropout(
            pool2, rate = drop_rate_2, training = training,
            seed = seed+4, name = "dropout_2")
    layers.append(("dropout_2", dropout2))

    # 2x conv3x3 relu batch-norm + pool
    conv5 = conv2d_relu_bn(
            dropout2, size = 3, n_filters = 128,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+5),
            name = "conv_5")
    layers.append(("conv_5", conv5))
    conv6 = conv2d_relu_bn(
            conv5, size = 3, n_filters = 128,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+6),
            name = "conv_6")
    layers.append(("conv_6", conv6))
    pool3 = max_pool2d(conv6, size = 2, stride = 2, name="pool_3")
    layers.append(("pool_3", pool3))
    dropout3 = tf.layers.dropout(
            pool3, rate = drop_rate_3, training = training,
            seed = seed+6, name = "dropout_3")
    layers.append(("dropout_3", dropout3))

    flat = flatten(dropout3, name="flatten")
    layers.append(("flatten", flat))
    
    # dense softmax
    dense1 = dense(
            flat, n_units = 10,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+7),
            name = "dense_1")
    layers.append(("logit", dense1))
    prob = tf.nn.softmax(dense1, name="prob")
    layers.append(("prob", prob))
    
    return layers, variables


# Springenberg et al. Striving for Simplicity: The All Convolutional Net
#https://arxiv.org/pdf/1412.6806.pdf
#https://github.com/MateLabs/All-Conv-Keras
#referred as ALL-CNN-C in the paper
def cifar10_sequential_allconvC_wd(
        x,
        drop_rate = 0.5,
        weight_decay = 0.001,
        seed = 42):
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name = "training")
    variables.append(("training", training))

    conv1 = conv2d_relu(
            x, size = 3, n_filters = 96,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+1),
            name = "conv_1")
    layers.append(("conv_1", conv1))    

    conv2 = conv2d_relu(
            conv1, size = 3, n_filters = 96,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2),
            name = "conv_2")
    layers.append(("conv_2", conv2))

    conv3 = conv2d_relu(
            conv2, stride = 2, size = 3, n_filters = 96,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+3),
            name = "conv_3")
    layers.append(("conv_3", conv3))
    
    dropout1 = tf.layers.dropout(
            conv3, rate = drop_rate, training = training,
            seed = seed+3, name = "dropout_1")
    layers.append(("dropout_1", dropout1))
    
    conv4 = conv2d_relu(
            dropout1, size = 3, n_filters = 192,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4),
            name = "conv_4")
    layers.append(("conv_4", conv4))

    conv5 = conv2d_relu(
            conv4, size = 3, n_filters = 192,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+5),
            name = "conv_5")
    layers.append(("conv_5", conv5))

    conv6 = conv2d_relu(
            conv5, stride = 2, size = 3, n_filters = 192,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+6),
            name = "conv_6")
    layers.append(("conv_6", conv6))

    dropout2 = tf.layers.dropout(
            conv6, rate = drop_rate, training = training,
            seed = seed+6, name = "dropout_2")
    layers.append(("dropout_2", dropout2))

    conv7 = conv2d_relu(
            dropout2, size = 3, n_filters = 192,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+7),
            name = "conv_7")
    layers.append(("conv_7", conv7))
    
    conv8 = conv2d_relu(
            conv7, size = 1, n_filters = 192,
            padding = "VALID",
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+8),
            name = "conv_8")
    layers.append(("conv_8", conv8))

    conv9 = conv2d_relu(
            conv8, size = 1, n_filters = 10,
            padding = "VALID",
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+9),
            name = "conv_9")
    layers.append(("conv_9", conv9))

    pool = global_avg_pool2d(conv9)    
    layers.append(("logit", pool))
    
    prob = tf.nn.softmax(pool, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables

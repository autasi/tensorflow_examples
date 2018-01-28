# -*- coding: utf-8 -*-

import tensorflow as tf
from arch.layers import conv2d_bn, global_avg_pool2d, dense
from arch.initializers import He_normal, Kumar_normal
import nasnet

def cifar10_nasnet(x, drop_rate = 0.0, seed = 42):
    penultimate_filters = 768
    nb_blocks = 6
    stem_filters = 32
    filters_multiplier = 2

    filters = penultimate_filters // 24 # 2x2x6 -> increase two times 2x and concatenate 6 branches
    
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))
    
    
    x = conv2d_bn(
        x, n_filters = stem_filters, size = 3, stride = 1,
        is_training = training,
        kernel_init = He_normal(seed = seed+1),
        name = "initial_conv")
    layers.append(("initial_conv", x))
    
    p = None
        
    for i in range(nb_blocks):
        x, p = nasnet.Normal_A(
            x, p,
            n_filters = filters,
            is_training = training,
            kernel_init = He_normal(seed = seed+2+i),
            name = "nasnet_normal_"+str(i)
            )
        layers.append(("nasnet_normal_"+str(i), x))

    x, _ = nasnet.Reduction_A(
        x, p,
        n_filters = filters * filters_multiplier,
        is_training = training,
        kernel_init = He_normal(seed = seed+3+nb_blocks),
        name = "nasnet_reduction_0"
        )
    layers.append(("nasnet_reduction_0", x))

    for i in range(nb_blocks):
        x, p = nasnet.Normal_A(
            x, p,
            n_filters = filters * filters_multiplier,
            is_training = training,
            kernel_init = He_normal(seed = seed+4+nb_blocks+i),
            name = "nasnet_normal_"+str(nb_blocks+i)
            )
        layers.append(("nasnet_normal_"+str(nb_blocks+i), x))

    x, _ = nasnet.Reduction_A(
        x, p,
        n_filters = filters * filters_multiplier ** 2,
        is_training = training,
        kernel_init = He_normal(seed = seed+5+2*nb_blocks),
        name = "nasnet_reduction_1"
        )
    layers.append(("nasnet_reduction_1", x))

    aux = nasnet.auxiliary_classifier(
        x, classes = 10,
        is_training = training,
        conv_kernel_init = He_normal(seed = seed),
        dense_kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed),
        name = "nasnet_aux_classifier"
        )
    layers.append(("aux_logit", aux))
    aux_prob = tf.nn.softmax(aux, name = "prob")
    layers.append(("aux_prob", aux_prob)) 

    for i in range(nb_blocks):
        x, p = nasnet.Normal_A(
            x, p,
            n_filters = filters * filters_multiplier ** 2,
            is_training = training,
            kernel_init = He_normal(seed = seed+6+2*nb_blocks+i),
            name = "nasnet_normal_"+str(2*nb_blocks+i)
            )
        layers.append(("nasnet_normal_"+str(2*nb_blocks+i), x))
        
    x = tf.nn.relu(x, name = "relu")    
    layers.append(("relu", x))
    
    x = global_avg_pool2d(x, name = "pool")
    layers.append(("pool", x))
    if drop_rate > 0.0:
        x = tf.layers.dropout(
                x, rate = drop_rate, training = training,
                seed = seed+7+3*nb_blocks, name = "dropout")
        layers.append(("dropout", x))
    x = dense(
            x, n_units = 10,
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+8+3*nb_blocks),
            name = "dense")
    layers.append(("logit", x))
    
    prob = tf.nn.softmax(x, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables
    
    
    
def cifar10_nasnet_wd(x, drop_rate = 0.0, weight_decay = 5e-4, seed = 42):
    penultimate_filters = 768
    nb_blocks = 6
    stem_filters = 32
    filters_multiplier = 2

    filters = penultimate_filters // 24 # 2x2x6 -> increase two times 2x and concatenate 6 branches
    
    layers = []
    variables = []

    training = tf.placeholder(tf.bool, name="training")
    variables.append(("training", training))
    
    
    x = conv2d_bn(
        x, n_filters = stem_filters, size = 3, stride = 1,
        is_training = training,
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
        kernel_init = He_normal(seed = seed+1),
        name = "initial_conv")
    layers.append(("initial_conv", x))
    
    p = None
        
    for i in range(nb_blocks):
        x, p = nasnet.Normal_A(
            x, p,
            n_filters = filters,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+2+i),
            name = "nasnet_normal_"+str(i)
            )
        layers.append(("nasnet_normal_"+str(i), x))

    x, _ = nasnet.Reduction_A(
        x, p,
        n_filters = filters * filters_multiplier,
        is_training = training,
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
        kernel_init = He_normal(seed = seed+3+nb_blocks),
        name = "nasnet_reduction_0"
        )
    layers.append(("nasnet_reduction_0", x))

    for i in range(nb_blocks):
        x, p = nasnet.Normal_A(
            x, p,
            n_filters = filters * filters_multiplier,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+4+nb_blocks+i),
            name = "nasnet_normal_"+str(nb_blocks+i)
            )
        layers.append(("nasnet_normal_"+str(nb_blocks+i), x))

    x, _ = nasnet.Reduction_A(
        x, p,
        n_filters = filters * filters_multiplier ** 2,
        is_training = training,
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
        kernel_init = He_normal(seed = seed+5+2*nb_blocks),
        name = "nasnet_reduction_1"
        )
    layers.append(("nasnet_reduction_1", x))

    aux = nasnet.auxiliary_classifier(
        x, classes = 10,
        is_training = training,
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
        conv_kernel_init = He_normal(seed = seed),
        dense_kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed),
        name = "nasnet_aux_classifier"
        )
    layers.append(("aux_logit", aux))
    aux_prob = tf.nn.softmax(aux, name = "prob")
    layers.append(("aux_prob", aux_prob)) 

    for i in range(nb_blocks):
        x, p = nasnet.Normal_A(
            x, p,
            n_filters = filters * filters_multiplier ** 2,
            is_training = training,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = He_normal(seed = seed+6+2*nb_blocks+i),
            name = "nasnet_normal_"+str(2*nb_blocks+i)
            )
        layers.append(("nasnet_normal_"+str(2*nb_blocks+i), x))
        
    x = tf.nn.relu(x, name = "relu")    
    layers.append(("relu", x))
    
    x = global_avg_pool2d(x, name = "pool")
    layers.append(("pool", x))
    if drop_rate > 0.0:
        x = tf.layers.dropout(
                x, rate = drop_rate, training = training,
                seed = seed+7+3*nb_blocks, name = "dropout")
        layers.append(("dropout", x))
    x = dense(
            x, n_units = 10,
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_init = Kumar_normal(activation = None, mode = "FAN_IN", seed = seed+8+3*nb_blocks),
            name = "dense")
    layers.append(("logit", x))
    
    prob = tf.nn.softmax(x, name = "prob")
    layers.append(("prob", prob))
    
    return layers, variables
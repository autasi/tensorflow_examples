#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import tensorflow as tf


def save_variables(session, path):
    """Saves tensorflow variables to file.
    Args:
        session: A TensorFlow session to retrieve the variables from.
        path: A string representing the path of the file.
    """    
    f = h5py.File(path, "w")
    for v in tf.trainable_variables():
        var_name = v.name
        with session.as_default():
            var_val = session.run(v)
        f[var_name] = var_val
    f.close()


def load_variables(session, path):
    """Loads tensorflow variables from file.
    Args:
        session: A TensorFlow session to retrieve the variables from.
        path: A string representing the path of the file.
    """    
    f = h5py.File(path, "r")
    for v in tf.trainable_variables():
        var_name = v.name
        var_val = f[var_name].value
        with session.as_default():
            session.run(v.assign(var_val))
    f.close()

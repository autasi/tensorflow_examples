#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2
from functools import partial

def translation_matrix(t):
    """Creates a translation transformation matrix.
    Args:
        t: A numpy array representing the translations along the x and y axes.
    Returns:
        A transformation matrix represented by a 3x3 numpy array.
    """     
    M = np.identity(3, dtype=np.float)
    M[:2, 2] = t
    return M


def rotation_matrix(angle):
    """Creates a rotation transformation matrix.
    Args:
        angle: A number representing the rotation in radians.
    Returns:
        A transformation matrix represented by a 3x3 numpy array.
    """    
    sa = math.sin(angle)
    ca = math.cos(angle)
    M = np.diag([ca, ca, 1.0])
    M += np.array([[0.0, -sa, 0.0],
                   [ sa, 0.0, 0.0],
                   [0.0, 0.0, 0.0]], dtype=np.float)
    return M


def scale_matrix(scale):
    """Creates a scaling transformation matrix.
    Args:
        scale: A number representing the ratio of scaling.
    Returns:
        A transformation matrix represented by a 3x3 numpy array.
    """    
    M = np.diag([scale, scale, 1.0])
    return M


def y_reflection_matrix():
    """Creates a reflection transformation matrix along y-axis.
    Returns:
        A transformation matrix represented by a 3x3 numpy array.
    """       
    M = np.diag([-1.0, 1.0, 1.0])
    return M


def x_reflection_matrix():
    """Creates a reflection transformation matrix along x-axis.
    Returns:
        A transformation matrix represented by a 3x3 numpy array.
    """       
    M = np.diag([1.0, -1.0, 1.0])
    return M


def transformation_matrix(transformations):
    """Combines multiple a transformation matrices into one.
    Args:
        transformations: A list of transformation matrices, each represented by
            a 3x3 numpy array.
    Returns:
        A transformation matrix represented by a 3x3 numpy array.
    """    
    if isinstance(transformations, np.ndarray):
        return transformations
    M = None
    for tr in transformations:
        if M is None:
            M = tr
        else:
            M = np.dot(tr, M)
    return M


class Affine(object):
    """Class to perform affine transformation on images.
    
    Attributes:
        shape: A tuple representing the shape of the input image.
        r: A number representing the rotation in degrees.
        tx: A number representing the translation along x-axis in pixels.
        ty: A number representing the translation along y-axis in pixels.
        scale: A number representing the ratio of scaling.
        reflect_y: A boolean flag to indicate the reflection in the y-axis.
        data_format: A string representing the data format, "channels_last" or
            "channels_first"
        tr_mat: An array representing the transformation matrix computed from
            the rotation, translation, scaling, and reflection paramters.

    """    
    def __init__(self, shape,
                       r=0.0, tx=0.0, ty=0.0, scale=1.0, reflect_y=False,
                       data_format="channels_last",
                       borderMode = cv2.BORDER_REPLICATE):
        self.shape = shape
        self.r = r
        self.tx = tx
        self.ty = ty
        self.scale = scale
        self.reflect_y = reflect_y
        self.data_format = data_format
        self.borderMode = borderMode
        self.tr_mat = None        
        self._init_tranformation_matrix()
        
    def _init_tranformation_matrix(self):
        """Initializes the affine transformation matrix using the rotation, 
        translation, scaling, and reflection paramters.
        """        
        if self.data_format == "channels_last":
            img_height = self.shape[0]
            img_width = self.shape[1]
        else:
            img_height = self.shape[1]
            img_width = self.shape[2]
        
        rotation = np.deg2rad(self.r)
        T = translation_matrix(np.array((self.tx, self.ty)))
        R = rotation_matrix(rotation)
        S = scale_matrix(self.scale)   
        if self.reflect_y:
            Fy = y_reflection_matrix() 
            self.tr_mat = transformation_matrix([Fy, S, R, T])
        else:
            self.tr_mat = transformation_matrix([S, R, T])

        center = 0.5*(np.array([img_width, img_height])-1)
        Tcenter = translation_matrix(-center)
        Torig = translation_matrix(center)
        self.tr_mat = transformation_matrix([Tcenter, self.tr_mat, Torig])
        
    
    def transform(self, x):
        """Applies affine transformation on input image.
        Args:
            x: A numpy array representing the input image.
        Returns:
            A numpy array representing the transformed image.
        """        
        warp = partial(cv2.warpAffine, M = self.tr_mat[:2, :],
                                       flags = (cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR),
                                       borderMode = self.borderMode)
        if self.data_format == "channels_last":
            nchan = self.shape[2]
        else:
            nchan = self.shape[0]
        xc = np.zeros_like(x, dtype=x.dtype)
        for i in range(0, nchan):
            if self.data_format == "channels_last":
                xc[:,:,i] = warp(x[:,:,i], dsize=x[:,:,i].shape)
            else:
                xc[i,:,:] = warp(x[i,:,:], dsize=x[i,:,:].shape)
        return xc
    

class RandomizedTransformer(object):
    """Class to create and apply transformations with randomized parameters.

    Attributes:
        transformer_class: The class of the transformation.
        parms: A list of tuples representing the input parameters. Each tuple
            contains the parameter name and its value.
        rand_parms: A list of tuples representing the random parameters. Each 
            tuple contains the parameter name and its random range. For ranges
            defined as a [low, high] list uniform sampling is performed. For 
            other cases (options) an item is randomly selected.
        mode: A string representing the mode how the tranformer is randomized.
            If its value "each" then upon each call a new random transformer
            is created. Otherwise, random transformer is created only once
            during the initialization.
        random_seed: A number representing the random seed.
        transformer: An instance of the transformer_class with randomized
            parameters.

    """      
    def __init__(self, transformer_class = None,
                       params = None,
                       rand_params = None,
                       mode = 'each',
                       random_seed = 42):
        
        self.transformer_class = transformer_class
        self.params = params
        self.rand_params = rand_params
        self.mode = mode
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        self.transformer = None
        if mode == 'once':
            self._init_random_transformer()
            
    def _init_random_transformer(self):
        """Initializes the random transformation.
        """         
        if (self.mode == 'once') and (self.transformer is not None):
            return
        all_params = dict()
        if self.rand_params is not None:
            for rand_param in self.rand_params:
                param_name = rand_param[0]
                param_range = rand_param[1]
                if param_range is None: # keep the default value
                    continue
                if isinstance(param_range, list): # random parameters
                    if isinstance(param_range[0], (bool, str)): # options
                        param_val = self.random_state.choice(param_range)
                    else:
                        if len(param_range) == 2: # range of [low, high)
                            param_val = self.random_state.uniform(*param_range)
                        else: # options
                            param_val = self.random_state.choice(param_range)
                else:
                    param_val = param_range
                all_params.update({param_name:param_val})
        if self.params is not None:
            all_params.update(dict(self.params))
        self.transformer = self.transformer_class(**all_params)
                
    def transform(self, x):
        """Applies random transformation on input data.
        Args:
            x: Input compatible with the input of the randomized transformer.
        Returns:
            Output compatible with the output of the randomized transformer.
        """        
        if self.mode == 'each':
            self._init_random_transformer()
        return self.transformer.transform(x)

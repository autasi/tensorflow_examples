# TensorFlow examples for training and testing
This repository contains some TensorFlow examples for training and using networks.
## Getting started
Follow the instructions below to get a working copy on your local machine.
### Prerequisites
The codes require the following libriaries:
* NumPy (http://www.numpy.org/)
* TensorFlow (https://www.tensorflow.org/)
* H5py (http://www.h5py.org/)

## Networks
The following networks are available:
* Basic ConvNets
  * Local response normalization
  * Batch normalalization
  * All convolutional network
* ResNet
  * Bottleneck blocks
* ResNeXt
* Inception networks
  * Inception v1
  * Batch normalized Inception v1
  * Inception v2
  * Inception v3
  * Inception v4
* Xception network
* DenseNet
* Squeeze and Excitation Networks
  * ResNet
  * ResNeXt
* MobileNet
* MobileNet V2 (in progress)
* ShuffleNet
* NASNet (in progress)

## Evaluations
Networks are evaluated against CIFAR-10 in two protocols using:
1. Basic - training with the fixed settings below
    * Data is normalized by global mean and std dev computed over *train* set
    * Randomly shuffled mini batches
    * Fixed 128 mini batch size
    * No data augmentation
    * Fixed 50 epochs
    * Optimizing the cross-entropy with AdamOptimizer
    * Learning rate with exponential decay, from 0.01 to 0.001
2. Author - training as done by the authors (except augmentation)
    * Data augmentation is fixed, namely using small random affine transformations (scale, rotation, and translation)

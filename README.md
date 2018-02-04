# TensorFlow examples for training and testing
This repository contains some TensorFlow examples of various networks architectures. Evaluations are performed on CIFAR-10.
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
1. **Basic** - training with the fixed settings as below
    * Data is normalized by global mean and std dev computed over *train* set
    * Randomly shuffled mini batches
    * Fixed 128 mini batch size
    * No data augmentation
    * No weight regularization applied
    * Training for fixed 50 epochs
    * Optimizing the cross-entropy with AdamOptimizer
    * Learning rate with exponential decay, from 0.01 to 0.001
2. **Author** - training as done by the authors (except augmentation)
    * Data augmentation is fixed, namely using small random affine transformations (scale, rotation, and translation)

## Results
### Networks
The list below contains the default settings for the networks:
* Convolution stride size is 1.
* Pooling stride size is 2.
* Kernel size for pooling is 2.
* Activation is ReLU, except the last dense layer, where SoftMax is used.
The following lists summarize the networks evaluated.
#### CN3D
1. Conv(size=5x5, filt=32) + LRN + MaxPool(size=3x3) + Dropout(0.2)
2. Conv(size=5x5, filt=64) + LRN + MaxPool(size=3x3) + Dropout(0.3)
3. Conv(size=3x3, filt=128) + LRN + MaxPool(size=2x2) + Dropout(0.3)
4. Dense
### Basic
Network
### Author

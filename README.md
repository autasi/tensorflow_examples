# TensorFlow network architectures and evaluations
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
  * Local response normalization (*LRN*)
  * Batch normalalization (*BN*)
  * All convolutional network
  * SELU activation and &alpha;-Dropout
* ResNet
  * Bottleneck blocks
* ResNeXt
* Inception networks
  * Inception v1
  * BN Inception v1
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
### Notations and settings
The list below contains the default settings for the networks. These are used by default in the network structures if not noted otherwise.
* Convolution stride size is 1.
* Convolution padding is 'same'. Valid will be denoted by *v*.
* Kernel size for pooling is 2x2.
* Pooling stride size is 2x2.
* Activation is ReLU, except the last dense layer, where SoftMax is used.
Moreover the *WD* term in the model name refers to the regularization applied on the weights, the regularization parameters is denoted by &lambda;. By default batch normalization (BN) is applied *after* the convolution and *before* the nonlinearities, and will be denoted by *Conv_BN*.
The following lists summarize the networks evaluated.
#### CLRN3D
1. Conv(5x5, 32) + LRN + MaxPool(3x3) + Dropout(0.2)
2. Conv(5x5, 64) + LRN + MaxPool(3x3) + Dropout(0.3)
3. Conv(3x3, 128) + LRN + MaxPool + Dropout(0.4)
4. Dense + SoftMax
#### CBN3D
Similar to **CN3D** however LRN is dropped and BN is applied. Everything else remains the same.
1. Conv_BN(5x5, 32) + MaxPool(3x3) + Dropout(0.2)
2. Conv_BN(5x5, 64) + MaxPool(3x3) + Dropout(0.3)
3. Conv_BN(3x3, 128) + MaxPool + Dropout(0.4)
4. Dense + SoftMax
#### CBN6D
This network uses blocks of Conv + ReLU + BN as
1. Conv(3x3, 32) + BN + Conv(3x3, 32) + BN + MaxPool + Dropout(0.2)
2. Conv(3x3, 64) + BN + Conv(3x3, 64) + BN + MaxPool + Dropout(0.3)
3. Conv(3x3, 128) + BN + Conv(3x3, 128) + BN + MaxPool + Dropout(0.4)
4. Dense + SoftMax
#### CBN6D-WD
The same as **CBN6D** with weight regularization parameter &lambda;=0.0001
#### AllConvC-WD
1. Dropout(0.2)
2. Conv(3x3, 96) + Conv(3x3, 96) + Conv(3x3, 96, stride=2x2) + Dropout(0.5)
3. Conv(3x3, 192) + Conv(3x3, 192) + Conv(3x3, 192, stride=2x2) + Dropout(0.5)
4. Conv(3x3, 192, v)
5. Conv(1x1, 192)
6. Conv(1x1, 10)
7. Global_AvgPool + SoftMax
#### ResNet-20
1. Conv_BN(3x3, 16)
2. Residual_layer(blocks=3, filters=16, stride=1)
3. Residual_layer(blocks=3, filters=32, stride=2)
4. Residual_layer(blocks=3, filters=64, stride=2)
5. Global_AvgPool
6. Dense + SoftMax
#### ResNet-32
1. Conv_BN(3x3, 16)
2. Residual_layer(blocks=5, filters=16, stride=1)
3. Residual_layer(blocks=5, filters=32, stride=2)
4. Residual_layer(blocks=5, filters=64, stride=2)
5. Global_AvgPool
6. Dense + SoftMax
#### ResNet-BN-20
The same as **ResNet-20** but using bottleneck blocks. It has more convolution layers, but i kept the *20* for simplicity.
#### ResNet-BN-32
The same as **ResNet-32** but using bottleneck blocks. It has more convolution layers, but i kept the *32* for simplicity.
#### ResNet-20-WD
The same as **ResNet-20** with weight regularization parameter &lambda;=0.0001
#### ResNet-ID-20
The same as **ResNet-20** but with the identity mapping structure.

### Basic
|     Network     | Accuracy |
|-----------------|----------|
|           CLRN3D|   0.5258 |
|            CBN3D|   0.8260 |
|         C3D-SELU|   0.6423 |
|    C3D-SELU-DROP|   0.6417 |
|            CBN6D|   0.8657 |
|        ResNet-20|   0.8208 |
|        ResNet-32|   0.8323 |
|     ResNet-BN-20|   0.8063 |
|     ResNet-BN-32|   0.8237 |
|     ResNet-IM-20|   0.8283 |
|       ResNeXt-29|   0.8718 |
|      DenseNet-40|   0.8794 |
|  DenseNet-BN-100|   0.8791 |
|     SE-ResNet-20|   0.8313 |
|    SE-ResNeXt-29|   0.8670 |
|     Inception-v1|   0.7676 |
|  BN-Inception-v1|   0.9005 |
|     Inception-v2|   0.8873 |
|     Inception-v3|   0.9244 |
|     Inception-v4|   0.8935 |
|         Xception|   0.8512 |
|        MobileNet|   0.8271 |
|       ShuffleNet|   0.8748 |
### Author

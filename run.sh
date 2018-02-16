#!/bin/bash

echo "resnet identity 20 wd"
python3 scripts/cifar10/cifar10_resnet_identity_20_wd.py

echo "xception"
python3 scripts/cifar10/cifar10_xception.py

echo "mobilenet"
python3 scripts/cifar10/cifar10_mobilenet.py

echo "shufflenet"
python3 scripts/cifar10/cifar10_shufflenet.py

echo "shufflenet wd"
python3 scripts/cifar10/cifar10_shufflenet_wd.py

#!/bin/bash

echo "bn inception v1 wd"
python3 scripts/cifar10/cifar10_bn_inception_v1_wd.py

echo "inception v2 wd"
python3 scripts/cifar10/cifar10_inception_v2_wd.py

echo "inception v3 wd"
python3 scripts/cifar10/cifar10_inception_v3_wd.py

echo "inception v4 wd"
python3 scripts/cifar10/cifar10_inception_v4_wd.py

echo "inception xception wd"
python3 scripts/cifar10/cifar10_xception_wd.py

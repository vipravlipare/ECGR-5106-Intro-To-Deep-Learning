# Homework 1 - CIFAR-10 CNN Experiments

This folder contains my Homework 1 work for ECGR 5106 Introduction to Deep Learning.

The homework uses PyTorch and CIFAR-10 to compare different CNN architectures:

- Problem 1: Modified AlexNet on CIFAR-10
- Problem 2: Modified VGGNet on CIFAR-10
- Problem 3: ResNet-11 vs ResNet-18 on CIFAR-10

## Files

- `Homework1_Problem1_Modified_AlexNet.ipynb`
- `Homework1_Problem2_Modified_VGGNet.ipynb`
- `Homework1_Problem3_ResNet_CIFAR10.ipynb`
- `Report/Homework 1 Report - ECGR 5106 - Viprav Lipare.pdf`

## Setup

The notebooks were run using PyTorch on CIFAR-10. The main settings used were:

- Random seed: 42
- Optimizer: AdamW
- Learning rate: 0.001
- Batch size: 256
- Scheduler: ReduceLROnPlateau
- Hardware: CPU

For faster laptop training, `fast_cpu=True` was used. This trains on a fixed 10,000 image training subset and a 2,000 image validation subset while keeping the full CIFAR-10 test set.

## Notes

The `Data/`, `Results/`, `Results_P1/`, `Results_P2/`, `Results_P3/`, and `Lectures/` folders are ignored by git because they contain downloaded data, saved model files, generated plots, or class materials. They can be recreated by running the notebooks.

# NeutronRobustness

This repository contains a simple code suite in Pytorch-Lightning to train our models.

main.py is the script used to launch the training phase, accepting different arguments such as dataset (CIFAR10 or
CIFAR100), network model, optimizer, learning rate, etc..

utils.py contains utilities for loading the dataset and building the network.

resnetCIFAR.py contains the ResNets architectures designed for the CIFAR dataset.

LightningModelWrapper.py contains a class which wraps a Pytorch model handling the training and validation loops and the
logging to wandb.

By default, logs are written to "wandb" directory, checkpoints to "checkpoints" directory and the datasets are
downloaded to "data" directory.

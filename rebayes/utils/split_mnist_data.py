
import re
import io
import os
import jax
import chex

import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from typing import Union
from jaxtyping import  Float, Array


import torchvision
from torchvision.transforms import ToTensor

import jax_dataloader.core as jdl


def load_mnist(root="./data", download=True):
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=download)
    images = np.array(mnist_train.data) / 255.0
    labels = mnist_train.targets

    mnist_test = torchvision.datasets.MNIST(root=root, train=False)
    images_test = np.array(mnist_test.data) / 255.0
    labels_test = mnist_test.targets

    train = (images, labels)
    test = (images_test, labels_test)
    return train, test



def make_split_mnist():
    # Modified from
    # https://github.com/probml/rebayes/blob/main/demos/showdown/split-mnist.ipynb
    train, test = load_mnist(root="./data", download=True)
    images, labels = train
    images_test, labels_test = test
    X_train, y_train = jnp.array(images), jnp.array(labels.ravel())
    X_test, y_test = jnp.array(images_test), jnp.array(labels_test.ravel())

    train_set_by_digit_pair, test_set_by_digit_pair = {}, {}
    for i in range(10//2):
        train_indx = (y_train == 2*i) | (y_train == 2*i+1)
        train_set_by_digit_pair[str(2*i)+str(2*i+1)] = (X_train[train_indx], y_train[train_indx]-2*i, y_train[train_indx])
        test_indx = (y_test == 2*i) | (y_test == 2*i+1)
        test_set_by_digit_pair[str(2*i)+str(2*i+1)] = (X_test[test_indx], y_test[test_indx]-2*i, y_test[test_indx])
    return train_set_by_digit_pair, test_set_by_digit_pair

def get_data_for_tasks(tasks, data_set_per_digit_pair, ndata_per_dist):
    Xs, Ys, Ls, Ids = [], [],  [], []
    for i in tasks:
        name = str(2*i)+str(2*i+1)
        Xi, Yi, Li = data_set_per_digit_pair[name]
        src_ndx = jnp.arange(ndata_per_dist)
        Xs.append(Xi[src_ndx])
        Ys.append(Yi[src_ndx])
        Ls.append(Li[src_ndx])
        identifiers = int(i) * jnp.ones(len(src_ndx), dtype=int)
        Ids.append(identifiers)
    X = jnp.concatenate(Xs)
    Y = jnp.concatenate(Ys)
    L = jnp.concatenate(Ls)
    Id = jnp.concatenate(Ids)
    return X, Y, L, Id

class SplitMNIST():
    def __init__(
            self,
            ntrain_per_task: int,
            ntest_per_task: int
    ):
        self.ntrain_per_task = ntrain_per_task
        self.ntest_per_task = ntest_per_task
        (self.train_set_by_digit_pair, self.test_set_by_digit_pair) = make_split_mnist()

    def get_training_data_all_tasks(self):
        ntasks = 5
        Xtr, Ytr, Ltr, Itr = get_data_for_tasks(np.arange(ntasks), self.train_set_by_digit_pair, self.ntrain_per_task)
        return Xtr, Ytr, Ltr, Itr
    
    def get_test_data_for_task(self, task):
        Xte, Yte, Ltr, Ite = get_data_for_tasks([task], self.test_set_by_digit_pair, self.ntest_per_task)
        return Xte, Yte, Ltr, Ite
    
    def get_test_data_for_seen_tasks(self, task):
        Xte, Yte, Lte, Ite = get_data_for_tasks(jnp.arange(task+1), self.test_set_by_digit_pair, self.ntest_per_task)
        return Xte, Yte, Lte, Ite
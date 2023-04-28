"""
Preprocessing and data augmentation for the datasets.
"""
import re
import io
import os
import jax
import chex
import zipfile
import requests
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import matplotlib.pyplot as plt

from typing import Union
from jaxtyping import  Float, Array

import torchvision
from torchvision.transforms import ToTensor

import jax_dataloader.core as jdl

@chex.dataclass
class LRState:
    params: Float[Array, "dim_input"]
    cov: Float[Array, "dim_input dim_input"]

class LRDataset:
    """
    L-RVGA's linear regression dataset
    Based on https://github.com/marc-h-lambert/L-RVGA
    """
    def __init__(self, dim_inputs, sigma, scale, condition_number, mean=None, rotate=True, normalize=False):
        self.dim_inputs = dim_inputs
        self.sigma = sigma
        self.scale = scale
        self.condition_number = condition_number
        self.rotate = rotate
        self.normalize = normalize
        self.mean = jnp.zeros(dim_inputs) if mean is None else mean
    
    def _normalize_if(self, normalize, array):
        if normalize:
            norm2 = jnp.linalg.norm(array) ** 2
            array = array / norm2
        return array
    
    def sample_covariance(self, key, normalize):
        diag = jnp.arange(1, self.dim_inputs + 1) ** self.condition_number
        diag = self.scale / diag
        diag = self._normalize_if(normalize, diag)
        
        cov = jnp.diag(diag)
        if self.dim_inputs > 1 and self.rotate:
            Q = jax.random.orthogonal(key, self.dim_inputs)
            cov = jnp.einsum("ji,jk,kl->il", Q, cov, Q)
        
        return cov
    
    def sample_inputs(self, key, mean, cov, n_obs):
        X = jax.random.multivariate_normal(key, mean, cov, (n_obs,))
        return X
    
    def sample_outputs(self, key, params, X):
        n_obs = len(X)
        err = jax.random.normal(key, (n_obs,))
        y = jnp.einsum("m,...m->...", params, X) + err * self.sigma
        return y
    
    def sample_train(self, key, num_obs):
        key_cov, key_x, key_params, key_y  = jax.random.split(key, 4)
        cov = self.sample_covariance(key_cov, self.normalize)
        
        params = jax.random.uniform(key_params, (self.dim_inputs,), minval=-1, maxval=1)
        params = params / jnp.linalg.norm(params)
        
        X = self.sample_inputs(key_x, self.mean, cov, num_obs)
        y = self.sample_outputs(key_y, params, X)
        
        state = LRState(
            params=params,
            cov=cov
        )
        
        return state, (X, y)
    
    def sample_test(self, key:jax.random.PRNGKey, state:LRState, num_obs:int):
        key_x, key_y = jax.random.split(key)
        X = self.sample_inputs(key_x, self.mean, state.cov, num_obs)
        y = self.sample_outputs(key_y, state.params, X)
        return X, y


def showdown_preprocess(
        train, test, n_warmup=1000, n_test_warmup=100, xaxis=0,
        normalise_target=True, normalise_features=True,
):
    (X_train, y_train) = train
    (X_test, y_test) = test

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    X_warmup = X_train[:n_warmup]
    y_warmup = y_train[:n_warmup]

    X_warmup_train = X_warmup[:-n_test_warmup]
    y_warmup_train = y_warmup[:-n_test_warmup]
    X_warmup_test = X_warmup[-n_test_warmup:]
    y_warmup_test = y_warmup[-n_test_warmup:]

    X_learn = X_train[n_warmup:]
    y_learn = y_train[n_warmup:]

    # Obtain mean and std of the warmup train set
    if normalise_target:
        ymean = y_warmup_train.mean().item()
        ystd = y_warmup_train.std().item()
    else:
        ymean, ystd = 0.0, 1.0
    

    if normalise_features:
        Xmean = X_warmup_train.mean(axis=xaxis, keepdims=True)
        Xstd = X_warmup_train.std(axis=xaxis, keepdims=True)
    else:
        Xmean, Xstd = 0.0, 1.0
    
    # Normalise input values
    X_warmup_train = (X_warmup_train - Xmean) / Xstd
    X_warmup_test = (X_warmup_test - Xmean) / Xstd
    X_learn = (X_learn - Xmean) / Xstd
    X_test = (X_test - Xmean) / Xstd
    # Normalise target values
    y_warmup_train = (y_warmup_train - ymean) / ystd
    y_warmup_test = (y_warmup_test - ymean) / ystd
    y_learn = (y_learn - ymean) / ystd
    y_test = (y_test - ymean) / ystd

    warmup_train = (X_warmup_train, y_warmup_train)
    warmup_test = (X_warmup_test, y_warmup_test)
    train = (X_learn, y_learn)
    test = (X_test, y_test)

    data = {
        "warmup_train": warmup_train,
        "warmup_test": warmup_test,
        "train": train,
        "test": test,
    }
    norm_cst = {
        "ymean": ymean,
        "ystd": ystd,
        "Xmean": Xmean,
        "Xstd": Xstd,
    }

    return data, norm_cst



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


def load_classification_mnist(
     root: str = "./data",
     num_train: int = 10_000,
):
    train, test = load_mnist(root=root)

    X, y = train
    X_test, y_test = test

    X = jnp.array(X)[:num_train].reshape(-1, 28 ** 2)
    y = jnp.array(y)[:num_train]
    y_ohe = jax.nn.one_hot(y, 10)

    X_test = jnp.array(X_test).reshape(-1, 28 ** 2)
    y_test = jnp.array(y_test)
    y_ohe_test = jax.nn.one_hot(y_test, 10)

    train = (X, y_ohe)
    test = (X_test, y_ohe_test)
    return train, test


def load_1d_synthetic_dataset(n_train=100, n_test=100, key=0, trenches=False, sort_data=False):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key1, key2, subkey1, subkey2, key_shuffle = jr.split(key, 5)

    n_train_sample = 2 * n_train if trenches else n_train
    X_train = jr.uniform(key1, shape=(n_train_sample, 1), minval=0.0, maxval=0.5)
    X_test = jr.uniform(key2, shape=(n_test, 1), minval=0.0, maxval=0.5)

    def generating_function(key, x):
        epsilons = jr.normal(key, shape=(3,))*0.02
        return (x + 0.3*jnp.sin(2*jnp.pi*(x+epsilons[0])) +
                0.3*jnp.sin(4*jnp.pi*(x+epsilons[1])) + epsilons[2])

    keys_train = jr.split(subkey1, X_train.shape[0])
    keys_test = jr.split(subkey2, X_test.shape[0])
    y_train = vmap(generating_function)(keys_train, X_train)
    y_test = vmap(generating_function)(keys_test, X_test)

    # Standardize dataset
    X_train = (X_train - X_train.mean()) / X_train.std()
    y_train = (y_train - y_train.mean()) / y_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    y_test = (y_test - y_test.mean()) / y_test.std()

    if trenches:
        sorted_idx = jnp.argsort(X_train.squeeze())
        train_idx = jnp.concatenate([
            sorted_idx[:n_train//2], sorted_idx[2*n_train - n_train//2:]
        ])

        X_train, y_train = X_train[train_idx], y_train[train_idx]

    if not sort_data:
        n_train = len(X_train)
        ixs = jr.choice(key_shuffle, shape=(n_train,), a=n_train, replace=False)
        X_train = X_train[ixs]
        y_train = y_train[ixs]
    else:
        sorted_idx = jnp.argsort(X_train.squeeze())
        X_train, y_train = X_train[sorted_idx], y_train[sorted_idx]

    return (X_train, y_train), (X_test, y_test)



def make_1d_regression(n_train=100, n_test=100, key=0, trenches=False, sort_data=False, coef=jnp.array([2.0,3.0]), sort_test=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key1, key2, subkey1, subkey2, key_shuffle = jr.split(key, 5)

    def gen(key, x):
        epsilons = jr.normal(key, shape=(3,))*0.02
        return (x + 0.3*jnp.sin(coef[0]*jnp.pi*(x+epsilons[0])) +
                0.3*jnp.sin(coef[1]*jnp.pi*(x+epsilons[1])) + epsilons[2])
    
    n_train_sample = 2 * n_train if trenches else n_train
    #X_train = jr.uniform(key1, shape=(n_train_sample, 1), minval=0.0, maxval=0.5)
    #X_test = jr.uniform(key2, shape=(n_test, 1), minval=0.0, maxval=0.5)
    X_train = jr.uniform(key1, shape=(n_train_sample, 1), minval=-0.5, maxval=0.5)
    X_test = jr.uniform(key2, shape=(n_test, 1), minval=-0.5, maxval=0.5)
    
    # sprt the test points for plotting 1d curves
    if sort_test:
        sorted_idx = jnp.argsort(X_test.squeeze())
        X_test = X_test[sorted_idx]

    keys_train = jr.split(subkey1, X_train.shape[0])
    keys_test = jr.split(subkey2, X_test.shape[0])
    y_train = vmap(gen)(keys_train, X_train)
    y_test = vmap(gen)(keys_test, X_test)

    # Standardize dataset
    X_train = (X_train - X_train.mean()) / X_train.std()
    y_train = (y_train - y_train.mean()) / y_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    y_test = (y_test - y_test.mean()) / y_test.std()

    if trenches:
        sorted_idx = jnp.argsort(X_train.squeeze())
        train_idx = jnp.concatenate([
            sorted_idx[:n_train//2], sorted_idx[2*n_train - n_train//2:]
        ])

        X_train, y_train = X_train[train_idx], y_train[train_idx]

    if not sort_data:
        n_train = len(X_train)
        ixs = jr.choice(key_shuffle, shape=(n_train,), a=n_train, replace=False)
        X_train = X_train[ixs]
        y_train = y_train[ixs]
    else:
        sorted_idx = jnp.argsort(X_train.squeeze())
        X_train, y_train = X_train[sorted_idx], y_train[sorted_idx]

    return X_train, y_train, X_test, y_test


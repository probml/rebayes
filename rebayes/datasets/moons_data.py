"""
Prepcocessing and data augmentation for the datasets.
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

from typing import Union
from jaxtyping import  Float, Array

from sklearn.datasets import make_moons


def make_showdown_moons(n_train, n_test, n_train_warmup, n_test_warmup, noise, seed=314):
    np.random.seed(seed)
    train = make_moons(n_samples=n_train, noise=noise)
    test = make_moons(n_samples=n_test, noise=noise)
    warmup_train = make_moons(n_samples=n_train_warmup, noise=noise)
    warmup_test = make_moons(n_samples=n_test_warmup, noise=noise)

    train = jax.tree_map(jnp.array, train)
    test = jax.tree_map(jnp.array, test)
    warmup_train = jax.tree_map(jnp.array, warmup_train)
    warmup_test = jax.tree_map(jnp.array, warmup_test)

    return train, test, warmup_train, warmup_test

def _rotation_matrix(angle):
    """
    Create a rotation matrix that rotates the
    space 'angle'-radians.
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return R


def _make_rotating_moons(radians, n_samples=100, **kwargs):
    """
    Make two interleaving half circles rotated by 'radians' radians
    
    Parameters
    ----------
    radians: float
        Angle of rotation
    n_samples: int
        Number of samples
    **kwargs:
        Extra arguments passed to the `make_moons` function
    """
    X, y = make_moons(n_samples=n_samples, **kwargs)
    X = jnp.einsum("nm,mk->nk", X, _rotation_matrix(radians))
    return X, y


def make_rotating_moons(n_train, n_test, n_rotations, min_angle=0, max_angle=360, seed=314, **kwargs):
    """
    n_train: int
        Number of training samples per rotation
    n_test: int
        Number of test samples per rotation
    n_rotations: int
        Number of rotations
    """
    np.random.seed(seed)
    n_samples = n_train + n_test
    min_rad = np.deg2rad(min_angle)
    max_rad = np.deg2rad(max_angle)

    radians = np.linspace(min_rad, max_rad, n_rotations)
    X_train_all, y_train_all, rads_train_all = [], [], []
    X_test_all, y_test_all, rads_test_all = [], [], []
    for rad in radians:
        X, y = _make_rotating_moons(rad, n_samples=n_samples, **kwargs)
        rads = jnp.ones(n_samples) * rad

        X_train = X[:n_train]
        y_train = y[:n_train]
        rad_train = rads[:n_train]

        X_test = X[n_train:]
        y_test = y[n_train:]
        rad_test = rads[n_train:]

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        rads_train_all.append(rad_train)

        X_test_all.append(X_test)
        y_test_all.append(y_test)
        rads_test_all.append(rad_test)

    X_train_all = jnp.concatenate(X_train_all, axis=0)
    y_train_all = jnp.concatenate(y_train_all, axis=0)
    rads_train_all = jnp.concatenate(rads_train_all, axis=0)
    X_test_all = jnp.concatenate(X_test_all, axis=0)
    y_test_all = jnp.concatenate(y_test_all, axis=0)
    rads_test_all = jnp.concatenate(rads_test_all, axis=0)

    train = (X_train_all, y_train_all, rads_train_all)
    test = (X_test_all, y_test_all, rads_test_all)
    
    return train, test


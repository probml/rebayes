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


from multiprocessing import Pool
from augly import image

import torchvision
from torchvision.transforms import ToTensor



class DataAugmentationFactory:
    """
    This is a base library to process / transform the elements of a numpy
    array according to a given function. To be used with gendist.TrainingConfig
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img, configs, n_processes=90):
        return self.process_multiple_multiprocessing(img, configs, n_processes)

    def process_single(self, X, *args, **kwargs):
        """
        Process a single element.

        Paramters
        ---------
        X: np.array
            A single numpy array
        kwargs: dict/params
            Processor's configuration parameters
        """
        return self.processor(X, *args, **kwargs)

    def process_multiple(self, X_batch, configurations):
        """
        Process all elements of a numpy array according to a list
        of configurations.
        Each image is processed according to a configuration.
        """
        X_out = []

        for X, configuration in zip(X_batch, configurations):
            X_processed = self.process_single(X, **configuration)
            X_out.append(X_processed)

        X_out = np.stack(X_out, axis=0)
        return X_out

    def process_multiple_multiprocessing(self, X_dataset, configurations, n_processes):
        """
        Process elements in a numpy array in parallel.

        Parameters
        ----------
        X_dataset: array(N, ...)
            N elements of arbitrary shape
        configurations: list
            List of configurations to apply to each element. Each
            element is a dict to pass to the processor.
        n_processes: [int, None]
            Number of cores to use. If None, use all available cores.
        """
        num_elements = len(X_dataset)
        if type(configurations) == dict:
            configurations = [configurations] * num_elements

        if n_processes == 1:
            dataset_proc = self.process_multiple(X_dataset, configurations)
            return dataset_proc.reshape(num_elements, -1)

        dataset_proc = np.array_split(X_dataset, n_processes)
        config_split = np.array_split(configurations, n_processes)
        elements = zip(dataset_proc, config_split)

        with Pool(processes=n_processes) as pool:
            dataset_proc = pool.starmap(self.process_multiple, elements)
            dataset_proc = np.concatenate(dataset_proc, axis=0)
        pool.join()

        return dataset_proc.reshape(num_elements, -1)


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


def rotate_mnist(X, angle):
    """
    Rotate an image by a given angle.
    We take the image to be a square of size 28x28.
    TODO: generalize to any size
    """
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))

    return X_shift

def generate_rotated_images(images, n_processes, minangle=0, maxangle=180):
    n_configs = len(images)
    processer = DataAugmentationFactory(rotate_mnist)
    angles = np.random.uniform(minangle, maxangle, n_configs)
    configs = [{"angle": float(angle)} for angle in angles]
    images_proc = processer(images, configs, n_processes=n_processes)
    return images_proc, angles


def load_rotated_mnist(
    root: str = "./data",
    target_digit: Union[int, None] = None,
    minangle: int = 0,
    maxangle: int = 180,
    n_processes: Union[int, None] = 1,
    num_train: int = 5_000,
    frac_train: Union[float, None] = None,
    seed: int = 314,
    sort_by_angle: bool = False,
):
    """
    """
    if seed is not None:
        np.random.seed(seed)

    if n_processes is None:
        n_processes = max(1, os.cpu_count() - 2)

    train, test = load_mnist(root=root)
    (X_train, labels_train), (X_test, labels_test) = train, test

    if target_digit is not None:
        digits = [target_digit] if type(target_digit) == int else target_digit

        map_train = [label in digits for label in labels_train]
        map_test = [label in digits for label in labels_test]
        X_train = X_train[map_train]
        X_test = X_test[map_test]

    X = np.concatenate([X_train, X_test], axis=0)
    (X, y) = generate_rotated_images(X, n_processes, minangle=minangle, maxangle=maxangle)

    X = jnp.array(X)
    y = jnp.array(y)

    if (frac_train is None) and (num_train is None):
        raise ValueError("Either frac_train or num_train must be specified.")
    elif (frac_train is not None) and (num_train is not None):
        raise ValueError("Only one of frac_train or num_train can be specified.")
    elif frac_train is not None:
        num_train = round(frac_train * len(X_train))

    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]

    if sort_by_angle:
        ix_sort = jnp.argsort(y_train)
        X_train = X_train[ix_sort]
        y_train = y_train[ix_sort]

    train = (X_train, y_train)
    test = (X_test, y_test)

    return train, test



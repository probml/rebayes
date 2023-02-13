"""
Prepcocessing and data augmentation for the datasets.
"""
import re
import io
import os
import jax
import zipfile
import requests
import torchvision
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from augly import image
from typing import Union
from multiprocessing import Pool


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
    num_train: int = 10_000,
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
        X_train = X_train[labels_train == target_digit]
        X_test = X_test[labels_test == target_digit]

    n_train = len(X_train)
    X = np.concatenate([X_train, X_test], axis=0)
    (X, y) = generate_rotated_images(X, n_processes, minangle=minangle, maxangle=maxangle)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)

    if num_train is not None:
        X_train = X_train[:num_train]
        y_train = y_train[:num_train]

    if sort_by_angle:
        ix_sort = jnp.argsort(y_train)
        X_train = X_train[ix_sort]
        y_train = y_train[ix_sort]

    train = (X_train, y_train)
    test = (X_test, y_test)

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


def normalise_dataset(data, target_variable, frac_train, seed):
    """
    Randomise a dataframe, normalise by column and transform to jax arrays
    """
    data = data.sample(frac=1.0, replace=False, random_state=seed)

    n_train = round(len(data) * frac_train)

    X_train = data.drop(columns=[target_variable]).iloc[:n_train].values
    y_train = data[target_variable].iloc[:n_train].values

    X_test = data.drop(columns=[target_variable]).iloc[n_train:].values
    y_test = data[target_variable].iloc[n_train:].values

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = jnp.maximum(std, 1e-8)

    mean_y = y_train.mean()
    std_y = y_train.std()

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    y_train = (y_train - mean_y) / std_y
    y_test = (y_test - mean_y) / std_y

    # Convert to jax arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test)

    train = (X_train, y_train)
    test = (X_test, y_test)
    return train, test



def load_uci_wine_regression(color="all", frac_train=0.8, include_color=False, seed=314, normalise=True):
    """
    https://archive.ics.uci.edu/ml/datasets/wine+quality
    """
    target_variable = "quality"
    url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{color}.csv"

    match color:
        case "red" | "white":
            colorv = 0 if color == "red" else 1
            url = url_base.format(color=color)
            data = (
                pd.read_csv(url, sep=";")
                .assign(color=colorv)
            )
        case "all":
            data_red = load_uci_wine_regression(color="red", include_color=True, normalise=False)
            data_white = load_uci_wine_regression(color="white", include_color=True, normalise=False)
            data = pd.concat([data_red, data_white], axis=0)

    if not include_color:
        data = data.drop(columns=["color"])

    if normalise:
        data = normalise_dataset(data, target_variable, frac_train, seed)

    return data


def load_uci_naval(target_variable="ship_speed", frac_train=0.8, seed=314):
    """
    http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants

    Target column is one of the following (default is "ship_speed"):
    * lever_position
    * ship_speed
    * gas_turbine_shaft_torque
    * gas_turbine_rate_of_revolutions
    * gas_generator_rate_of_revolutions
    * starboard_propeller_torque
    * port_propeller_torque
    * hp_turbine_exit_temperature
    * gt_compressor_inlet_air_temperature
    * gt_compressor_outlet_air_temperature
    * hp_turbine_exit_pressure
    * gt_compressor_inlet_air_pressure
    * gt_compressor_outlet_air_pressure
    * gas_turbine_exhaust_gas_pressure
    * turbine_injecton_control
    * fuel_flow
    * gt_compressor_decay_state_coefficient
    * gt_turbine_decay_state_coefficient
    """
    file_target = "UCI CBM Dataset/data.txt"
    file_features = "UCI CBM Dataset/Features.txt"
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip"

    r = requests.get(url)
    file = io.BytesIO(r.content)
    with zipfile.ZipFile(file) as zip:
        with zip.open(file_features) as f:
            features = f.read().decode("utf-8")
            regexp = re.compile("[0-9]{1,2} - ([\w\s]+)")
            features = regexp.findall(features)
            features = [f.lower().rstrip().replace(" ", "_") for f in features]

        with zip.open(file_target) as f:
            data = f.read().decode("utf-8")
            data = io.StringIO(data)
            data = pd.read_csv(data, sep="\s+", header=None, engine="python")
            data.columns = features

    data = normalise_dataset(data, target_variable, frac_train, seed)
    return data


def load_uci_kin8nm():
    ...


def load_uci_power():
    ...


def load_uci_protein():
    ...


def load_uci_spam():
    ...

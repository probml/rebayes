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

from multiprocessing import Pool
from augly import image

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


def normalise_dataset(data, target_variable, frac_train, seed, feature_normalise=False, target_normalise=False):
    #TODO: rename to uci_preprocess
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

    if feature_normalise:
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if target_normalise:
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


    if (color == "red") | (color == "white"):
        colorv = 0 if color == "red" else 1
        url = url_base.format(color=color)
        data = (
            pd.read_csv(url, sep=";")
            .assign(color=colorv)
        )
    elif color == "all":
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


def load_uci_kin8nm(frac_train=0.8, seed=314):
    url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"
    target_variable = "y"
    data = pd.read_csv(url)

    data = normalise_dataset(data, target_variable, frac_train, seed)    
    return data


def load_uci_power(frac_train=0.8, seed=314):
    """
    https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    file_target = "CCPP/Folds5x2_pp.xlsx"
    target_variable = "PE"

    r = requests.get(url)
    file = io.BytesIO(r.content)
    with zipfile.ZipFile(file) as zip:
        with zip.open(file_target) as f:
            data = io.BytesIO(f.read())
            data = pd.read_excel(data)
        
    data = normalise_dataset(data, target_variable, frac_train, seed)
    return data


def load_uci_protein(frac_train=0.8, seed=314):
    """
    https://github.com/yaringal/DropoutUncertaintyExps/tree/master/UCI_Datasets/protein-tertiary-structure
    """
    target_variable = 9
    url = (
        "https://raw.githubusercontent.com/yaringal/"
        "DropoutUncertaintyExps/master/UCI_Datasets/"
        "protein-tertiary-structure/data/data.txt"
    )
    data = pd.read_csv(url, header=None, sep=" ")
    data = normalise_dataset(data, target_variable, frac_train, seed)
    return data


def load_uci_spam(frac_train=0.8, seed=314):
    """
    https://archive.ics.uci.edu/ml/datasets/spambase
    """
    target_variable = 57
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    data = pd.read_csv(url, header=None)
    data = normalise_dataset(data, target_variable, frac_train, seed, target_normalise=False)
    return data


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


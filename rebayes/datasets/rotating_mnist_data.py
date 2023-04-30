"""
Prepcocessing and data augmentation for the datasets.
"""
from multiprocessing import Pool
import os
from typing import Callable, Tuple, Union

import augmax
from augmax.geometric import GeometricTransformation, LazyCoordinates
import torchvision
import numpy as np
import jax.numpy as jnp
import jax.random as jr


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
    
    
class Rotate(GeometricTransformation):
    """Rotates the image by a random arbitrary angle.

    Args:
        angle_range (float, float): Tuple of `(min_angle, max_angle)` to sample from.
            If only a single number is given, angles will be sampled from `(-angle_range, angle_range)`.
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            angle_range: Union[Tuple[float, float], float]=(-30, 30),
            p: float = 1.0):
        super().__init__()
        if not hasattr(angle_range, '__iter__'):
            angle_range = (-angle_range, angle_range)
        self.theta_min, self.theta_max = map(jnp.radians, angle_range)
        self.probability = p

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: LazyCoordinates, invert=False):
        do_apply = jr.bernoulli(rng, self.probability)
        theta = do_apply * jr.uniform(rng, minval=self.theta_min, maxval=self.theta_max)

        if invert:
            theta = -theta

        transform = jnp.array([
            [ jnp.cos(theta), jnp.sin(theta), 0],
            [-jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1]
        ])
        coordinates.push_transform(transform)


def load_mnist(root="/tmp/data", download=True):
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
    X_rot = X.reshape(28, 28)
    rotate_transform = augmax.Chain(
        Rotate((angle, angle,))
    )
    X_rot = rotate_transform(jr.PRNGKey(0), X_rot).reshape(X.shape)
    
    return X_rot


def generate_rotated_images(images, n_processes, minangle=0, maxangle=180, anglefn=None):
    n_configs = len(images)
    angles = anglefn(n_configs, minangle, maxangle)

    processer = DataAugmentationFactory(rotate_mnist)
    configs = [{"angle": float(angle)} for angle in angles]
    images_proc = processer(images, configs, n_processes=n_processes)
    return images_proc, angles


def generate_rotated_images_pairs(images, angles, n_processes=1):
    processer = DataAugmentationFactory(rotate_mnist)
    configs = [{"angle": float(angle)} for angle in angles]
    images_proc = processer(images, configs, n_processes=n_processes)
    return images_proc, angles


def load_rotated_mnist(
    anglefn: Callable,
    root: str = "/tmp/data",
    target_digit: Union[int, None] = None,
    minangle: int = 0,
    maxangle: int = 180,
    n_processes: Union[int, None] = 1,
    num_train: int = 5_000,
    frac_train: Union[float, None] = None,
    seed: int = 314,
    sort_by_angle: bool = False,
    num_test: int = None,
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

        digits_train = labels_train[map_train]
        digits_test = labels_test[map_test]
    else:
        digits_train = labels_train
        digits_test = labels_test

    X = np.concatenate([X_train, X_test], axis=0)
    digits = np.concatenate([digits_train, digits_test], axis=0)

    (X, y) = generate_rotated_images(X, n_processes, minangle=minangle, maxangle=maxangle, anglefn=anglefn)

    X = jnp.array(X)
    y = jnp.array(y)

    if (frac_train is None) and (num_train is None):
        raise ValueError("Either frac_train or num_train must be specified.")
    elif (frac_train is not None) and (num_train is not None):
        raise ValueError("Only one of frac_train or num_train can be specified.")
    elif frac_train is not None:
        num_train = round(frac_train * len(X_train))

    X_train, y_train, digits_train = X[:num_train], y[:num_train], digits[:num_train]
    if num_test is not None:
        X_test, y_test, digits_test = X[num_train : num_train + num_test], y[num_train : num_train + num_test], digits[num_train : num_train + num_test]
    else:
        X_test, y_test, digits_test = X[num_train:], y[num_train:], digits[num_train:]

    if sort_by_angle:
        ix_sort = jnp.argsort(y_train)
        X_train = X_train[ix_sort]
        y_train = y_train[ix_sort]
        digits_train = digits_train[ix_sort]

    train = (X_train, y_train, digits_train)
    test = (X_test, y_test, digits_test)

    return train, test


def load_and_transform(
    anglefn: Callable,
    digits: list,
    num_train: int = 5_000,
    frac_train: Union[float, None] = None,
    sort_by_angle: bool = True,
):
    """
    Function to load and transform the rotated MNIST dataset.
    """
    data = load_rotated_mnist(
        anglefn, target_digit=digits, sort_by_angle=sort_by_angle,
        num_train=num_train, frac_train=frac_train
    )
    train, test = data
    X_train, y_train, labels_train = train
    X_test, y_test, labels_test = test

    ymean, ystd = y_train.mean().item(), y_train.std().item()

    if ystd > 0:
        y_train = (y_train - ymean) / ystd
        y_test = (y_test - ymean) / ystd

    dataset = {
        "train": (X_train, y_train, labels_train),
        "test": (X_test, y_test, labels_test),
    }

    res = {
        "dataset": dataset,
        "ymean": ymean,
        "ystd": ystd,
    }

    return res

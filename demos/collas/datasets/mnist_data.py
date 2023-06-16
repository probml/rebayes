from functools import partial
from typing import Callable, Tuple

import augmax
import numpy as np
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
import tensorflow_datasets as tfds

from rebayes.datasets.data_utils import Rotate


# Helper Functions -------------------------------------------------------------

def _process_mnist(
    dataset: Tuple,
    n: int=None,
    key: int=0,
    shuffle: bool=True,
    oh: bool=True,
) -> Tuple:
    """Process a single element.
    """
    X, *args, Y = dataset
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    Y = jax.nn.one_hot(Y, 10) if oh else Y
    idx = jr.permutation(key, jnp.arange(len(X))) if shuffle \
        else jnp.arange(len(X))
    X, Y = X[idx], Y[idx]
    if n is not None:
        X, Y = X[:n], Y[:n]
    new_args = []
    for arg in args:
        if isinstance(arg, dict):
            arg = tree_map(lambda x: x[idx], arg)
        else:
            arg = arg[idx]
        new_args.append(arg)
            
    return X, *new_args, Y


def process_mnist_dataset(
    train: Tuple,
    val: Tuple,
    test: Tuple,
    ntrain: int=None,
    nval: int=None,
    ntest: int=None,
    key: int=0,
    shuffle: bool=True,
    oh_train: bool=True,
) -> dict:
    """Wrap MNIST dataset into a dictionary.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 3)
    train, val, test = \
        (_process_mnist(dataset, n, key, shuffle, oh)
         for dataset, n, key, oh in zip([train, val, test], 
                                        [ntrain, nval, ntest],
                                        keys, [oh_train, False, False]))
    dataset = {
        'train': train,
        'val': val,
        'test': test,
    }
    
    return dataset


# MNIST ------------------------------------------------------------------------

def load_mnist_dataset(
    data_dir: str="/tmp/data",
    fashion: bool=False,
    **process_kwargs,
) -> dict:
    """Load MNIST train, validatoin, and test datasets into memory.
    """
    dataset='mnist'
    if fashion:
        dataset='fashion_mnist'
    ds_builder = tfds.builder(dataset, data_dir=data_dir)
    ds_builder.download_and_prepare()
    
    train_ds, val_ds, test_ds = \
        (tfds.as_numpy(ds_builder.as_dataset(split=split, batch_size=-1)) 
         for split in ['train[10%:]', 'train[:10%]', 'test'])
    
    # Normalize pixel values
    for ds in [train_ds, val_ds, test_ds]:
        ds['image'] = np.float32(ds['image']) / 255.
    
    train, val, test = \
        ((jnp.array(ds['image']), jnp.array(ds['label'])) 
         for ds in [train_ds, val_ds, test_ds])
    
    dataset = process_mnist_dataset(train, val, test, **process_kwargs)

    return dataset


# Rotated MNIST ----------------------------------------------------------------

def generate_random_angles(
    n_tasks: int,
    min_angle: float=0.0,
    max_angle: float=180.0,
    key: int=0,
) -> jnp.ndarray:
    """Generate random angles.
    """
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)    
    angles = jr.uniform(key, (n_tasks,), minval=min_angle, maxval=max_angle)
    
    return angles


def rotate_mnist(
    img: jnp.ndarray,
    angle: float,
) -> jnp.ndarray:
    """Rotate MNIST image given an angle.
    """
    img_rot = img.reshape(28, 28)
    rotate_transform = augmax.Chain(
        Rotate((angle, angle,))
    )
    img_rot = rotate_transform(jr.PRNGKey(0), img_rot).reshape(img.shape)
    
    return img_rot


def process_angles(
    angles: jnp.ndarray,
) -> dict:
    """Process angles.
    """
    angles_mean, angles_std = angles.mean(), angles.std()
    angles_normalized = (angles - angles_mean) / angles_std
    
    return {
        "angles": angles,
        "angles_normalized": angles_normalized,
        "angles_mean": angles_mean,
        "angles_std": angles_std,
    }


def generate_rotated_images(
    imgs: jnp.ndarray,
    labels: jnp.ndarray,
    n: int=None,
    key: int=0,
    target_digit: int=None,
    angle_fn: Callable=None,
    min_angle: float=0.0,
    max_angle: float=180.0,
    include_labels: bool=True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate rotated images.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if angle_fn is None:
        angle_fn = generate_random_angles
    n = len(imgs) if n is None else min(n, len(imgs))
    if target_digit is not None:
        imgs = imgs[labels == target_digit][:n]
        labels = labels[labels == target_digit][:n]
    angles = angle_fn(len(imgs), min_angle, max_angle, key)
    imgs_rot = vmap(rotate_mnist)(imgs, angles)
    if include_labels:
        return imgs_rot, angles, labels
    
    return imgs_rot, angles


def load_rotated_mnist_dataset(
    dataset: dict=None,
    data_dir: str="/tmp/data",
    fashion: bool=False,
    key: int=0,
    target_digit: int=None,
    angle_fn: Callable=None,
    min_angle: float=0.0,
    max_angle: float=180.0,
    ntrain: int=None,
    nval: int=None,
    ntest: int=None,
    include_labels: bool=True,
    **process_kwargs,
) -> dict:
    """Load rotated MNIST dataset.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if dataset is None:
        dataset = load_mnist_dataset(data_dir, fashion, oh_train=False)
    train, val, test = \
        (generate_rotated_images(*dataset[k], n, key_, target_digit, angle_fn,
                                 min_angle, max_angle, include_labels) 
         for k, n, key_ in zip(['train', 'val', 'test'], [ntrain, nval, ntest],
                               jr.split(key, 3)))
    oh_train = True if include_labels else False
    dataset = process_mnist_dataset(train, val, test, oh_train=oh_train,
                                    **process_kwargs)
    
    return dataset
    
    
def load_seq_digits_rotated_mnist_dataset(
    ntrain_per_task: int,
    nval_per_task: int,
    ntest_per_task: int,
    data_dir: str="/tmp/data",
    fashion: bool=False,
    key: int=0,
    angle_fn: Callable=None,
    min_angle=0,
    max_angle=180,
) -> dict:
    """Load rotated MNIST dataset where each digit is used sequentially.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    datasets = []
    for i in range(10):
        dataset = load_rotated_mnist_dataset(
            data_dir, fashion, key, i, angle_fn, min_angle, max_angle,
            ntrain=ntrain_per_task, nval=nval_per_task, ntest=ntest_per_task,
        )
        datasets.append(dataset)
    dataset = tree_map(
        lambda *args: jnp.concatenate(args, axis=0),
        *datasets,
    )
    
    return dataset


# Permuted MNIST ---------------------------------------------------------------

def _permute(
    img: jnp.ndarray,
    idx: jnp.ndarray,
) -> jnp.ndarray:
    """Permute an image.
    """
    img_permuted = img.ravel()[idx].reshape(img.shape)
    
    return img_permuted


def _permute_dataset(
    imgs: jnp.ndarray,
    labels: jnp.ndarray,
    perm_idx: jnp.ndarray,
    n: int=None,
    key: int=0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    # Shuffle dataset
    n = len(imgs) if n is None else min(n, len(imgs))
    idx = jr.permutation(key, jnp.arange(len(imgs)))
    imgs, labels = imgs[idx][:n], labels[idx][:n]
    
    # Permute images
    imgs_permuted = vmap(_permute, (0, None))(imgs, perm_idx)
    
    return imgs_permuted, labels


def load_single_permuted_mnist_dataset(
    perm_idx: jnp.ndarray,
    ntrain: int,
    nval: int,
    ntest: int,
    dataset: dict=None,
    data_dir: str="/tmp/data",
    fashion: bool=False,
    key: int=0,
    **process_kwargs,
) -> dict:
    """Load permuted MNIST dataset given permuted index.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if dataset is None:
        dataset = load_mnist_dataset(data_dir, fashion, oh_train=False)
    keys = jr.split(key, 3)
        
    train, val, test = \
        (_permute_dataset(*dataset[k], perm_idx, n, key_) for k, n, key_ in 
         zip(['train', 'val', 'test'], [ntrain, nval, ntest], keys))
    
    dataset = process_mnist_dataset(train, val, test, **process_kwargs)
    
    return dataset


def load_permuted_mnist_dataset(
    n_tasks: int,
    ntrain_per_task: int,
    nval_per_task: int,
    ntest_per_task: int,
    data_dir: str="/tmp/data",
    fashion: bool=False,
    key: int=0,
    **process_kwargs,
) -> dict:
    """Load Permuted MNIST dataset.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    dataset = load_mnist_dataset(data_dir, fashion, oh_train=False)
    identity_idx = jnp.arange(28*28)[None, :]
    keys = jr.split(key, n_tasks)
    perm_idx = jnp.concatenate(
        [identity_idx, jnp.array([jr.permutation(keys[i], jnp.arange(28*28))
                                  for i in range(n_tasks-1)]),],
        axis=0,
    )
    keys = jr.split(keys[-1], n_tasks)
    permute_fn = lambda idx, k: \
        load_single_permuted_mnist_dataset(idx, ntrain_per_task, nval_per_task,
                                           ntest_per_task, dataset, key=k,
                                           **process_kwargs)
    dataset = vmap(permute_fn)(perm_idx, keys)
    # Squash vmapped axis
    dataset = tree_map(lambda x: jnp.concatenate(x, axis=0), dataset)
    
    return dataset


# Rotating Permuted MNIST ------------------------------------------------------

def load_rotated_permuted_mnist_dataset(
    n_tasks: int,
    ntrain_per_task: int,
    nval_per_task: int,
    ntest_per_task: int,
    data_dir: str="/tmp/data",
    fashion: bool=False,
    key: int=0,
    angle_fn: Callable=None,
    min_angle=0,
    max_angle=180,
) -> dict:
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 2)
    permuted_dataset = \
        load_permuted_mnist_dataset(n_tasks, ntrain_per_task, nval_per_task, 
                                    ntest_per_task, data_dir, fashion, keys[0],
                                    oh_train=False)
    dataset = load_rotated_mnist_dataset(
        permuted_dataset, angle_fn=angle_fn, min_angle=min_angle, 
        max_angle=max_angle, key=keys[1], shuffle=False,
    )
    
    return permuted_dataset, dataset


# Split MNIST ------------------------------------------------------------------

def make_split_mnist(
    imgs: jnp.ndarray,
    labels: jnp.ndarray,
    n: int=None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split given dataset into 5 tasks.
    """
    n_tasks = 10//2
    n = n or len(imgs)
    imgs_split, labels_split = [], []
    for i in range(n_tasks):
        task_idx = (labels == 2*i) | (labels == 2*i+1)
        imgs_curr, labels_curr = imgs[task_idx], labels[task_idx]
        r_idx = jr.permutation(jr.PRNGKey(i), len(imgs_curr))
        imgs_curr, labels_curr = imgs_curr[r_idx][:n], labels_curr[r_idx][:n]
        imgs_split.append(imgs_curr)
        labels_split.append(labels_curr - 2*i)
    imgs_split, labels_split = \
        jnp.concatenate(imgs_split), jnp.concatenate(labels_split)
        
    return imgs_split, labels_split


def load_split_mnist_dataset(
    ntrain_per_task: int, 
    nval_per_task: int, 
    ntest_per_task: int,
    data_dir: str="/tmp/data",
    fashion: bool=False,
    key=0,
) -> dict:
    """Load Split MNIST dataset, each with 2 consecutive digits given dataset.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    dataset = load_mnist_dataset(data_dir, fashion=fashion, oh_train=False)
    train, val, test = \
        (make_split_mnist(*dataset[k], n)
         for k, n in zip(('train', 'val', 'test'),
                         (ntrain_per_task, nval_per_task, ntest_per_task)))
    dataset = process_mnist_dataset(train, val, test, shuffle=False,
                                    oh_train=False)
    
    return dataset


# For experiments --------------------------------------------------------------

mnist_kwargs = {
    "ntrain": 500,
    "nval": 1_000,
}
pmnist_kwargs = {
    'n_tasks': 10,
    'ntrain_per_task': 300,
    'nval_per_task': 1,
    'ntest_per_task': 50,
}
smnist_kwargs = {
    'ntrain_per_task': 300,
    'nval_per_task': 1,
    'ntest_per_task': 500,
}
rmnist_kwargs = {
    "ntrain": 5_000,
    "nval": 100,
}

Datasets = {
    'stationary-mnist': {
        "load_fn": partial(load_mnist_dataset, **mnist_kwargs),
        "configs": mnist_kwargs,
    },
    'permuted-mnist': {
        "load_fn": partial(load_permuted_mnist_dataset, **pmnist_kwargs),
        "configs": pmnist_kwargs,
    },
    'rotated-mnist': {
        "load_fn": partial(load_rotated_mnist_dataset, include_labels=False,
                           **rmnist_kwargs),
        "configs": rmnist_kwargs,
    },
    'rotated-permuted-mnist': {
        "load_fn": load_rotated_permuted_mnist_dataset,
        "configs": {}
    },
    'split-mnist': {
        "load_fn": partial(load_split_mnist_dataset, **smnist_kwargs),
        "configs": smnist_kwargs,
    }
}
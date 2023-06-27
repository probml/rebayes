from functools import partial
from typing import Callable, Tuple

import augmax
import numpy as np
import jax
from jax import vmap
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
import tensorflow_datasets as tfds

from rebayes.datasets.data_utils import Rotate


# Helper Functions -------------------------------------------------------------

def _process(
    dataset: Tuple,
    n: int=None,
    key: int=0,
    shuffle: bool=True,
    oh: bool=True,
    output_dim: int=10,
) -> Tuple:
    """Process a single element.
    """
    X, *args, Y = dataset
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    Y = jax.nn.one_hot(Y, output_dim) if oh else Y
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


def process_dataset(
    train: Tuple,
    val: Tuple,
    test: Tuple,
    ntrain: int=None,
    nval: int=None,
    ntest: int=None,
    key: int=0,
    shuffle: bool=True,
    oh_train: bool=True,
    output_dim: int=10,
) -> dict:
    """Wrap dataset into a dictionary.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 3)
    train, val, test = \
        (_process(dataset, n, key, shuffle, oh, output_dim)
         for dataset, n, key, oh in zip([train, val, test], 
                                        [ntrain, nval, ntest],
                                        keys, [oh_train, False, False]))
    dataset = {
        'train': train,
        'val': val,
        'test': test,
    }
    
    return dataset


# Base Dataset -----------------------------------------------------------------

def load_base_dataset(
    data_dir: str="/tmp/data",
    dataset_type: str='fashion_mnist',
    **process_kwargs,
) -> dict:
    """Load train, validatoin, and test datasets into memory.
    """
    ds_builder = tfds.builder(dataset_type, data_dir=data_dir)
    ds_builder.download_and_prepare()
    
    train_ds, val_ds, test_ds = \
        (tfds.as_numpy(ds_builder.as_dataset(split=split, batch_size=-1)) 
         for split in ['train[10%:]', 'train[:10%]', 'test'])
    
    # Normalize pixel values
    for ds in [train_ds, val_ds, test_ds]:
        ds['image'] = np.float32(ds['image']) / 255.
    
    output_dim = train_ds["label"].max() + 1
    train, val, test = \
        ((jnp.array(ds['image']), jnp.array(ds['label'])) 
         for ds in [train_ds, val_ds, test_ds])
    
    dataset = process_dataset(train, val, test, output_dim=output_dim,
                              **process_kwargs)

    return dataset


# Rotated Dataset --------------------------------------------------------------

def generate_random_angles(
    n_tasks: int,
    min_angle: float=0.0,
    max_angle: float=180.0,
    key: int=0,
) -> jnp.ndarray:
    """Generate iid angles.
    """
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)    
    angles = jr.uniform(key, (n_tasks,), minval=min_angle, maxval=max_angle)
    
    return angles


def generate_amplified_angles(
    n_tasks: int,
    min_angle: float=0.0,
    max_angle: float=180.0,
    key: int=0,
) -> jnp.ndarray:
    """Generate angles with gradually increasing amplitude.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    t = jnp.linspace(0, 1.5, n_tasks)
    angles = jnp.exp(t) * jnp.sin(35 * t)
    angles /= angles.max()
    angles = (angles + 1) / 2 * (max_angle - min_angle) + \
        min_angle + jr.normal(key=key, shape=(n_tasks,)) * 2
        
    return angles


def generate_random_walk_angles(
    n_tasks: int,
    min_angle: float=0.0,
    max_angle: float=180.0,
    key: int=0,
    theta: float=10.0,
) -> jnp.ndarray:
    """Generate random walk angles using Ornstein-Uhlenbeck process.
    The gamma (decay) term corresponds to (1 - theta*dt), which, using
    n_tasks=2000 and theta=10.0, corresponds to 0.995.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    mean_angle = (min_angle+max_angle)/2
    std_angle = mean_angle/3
    dt = 1/n_tasks

    def _step(carry, key):
        prev_angle = carry
        next_angle = prev_angle + theta * dt * (mean_angle - prev_angle) + \
            std_angle * jnp.sqrt(2 * dt * theta) * jr.normal(key,)
        
        return next_angle, next_angle
    
    keys = jr.split(key, n_tasks)
    _, angles = scan(_step, mean_angle, keys)

    return angles


def rotate_image(
    img: jnp.ndarray,
    angle: float,
) -> jnp.ndarray:
    """Rotate image given an angle.
    """
    img_rot = img.squeeze()
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
    

def _filter_target_digit(
    dataset: Tuple,
    target_digit: int,
    n: int=None,
) -> Tuple:
    """Filter dataset for a target digit.
    """
    if target_digit == -1:
        return dataset
    
    X, *args, Y = dataset
    idx = Y == target_digit
    X, Y = X[idx], Y[idx]
    if n is not None:
        X, Y = X[:n], Y[:n]
    new_args = []
    for arg in args:
        if isinstance(arg, dict):
            arg = tree_map(lambda x: x[idx][:n], arg)
        else:
            arg = arg[idx][:n]
        new_args.append(arg)
            
    return X, *new_args, Y


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
    angles: jnp.ndarray=None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate rotated images.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if angle_fn is None:
        angle_fn = generate_random_angles
    if target_digit is not None:
        imgs = imgs[labels == target_digit]
        labels = labels[labels == target_digit]
    n = len(imgs) if n is None else min(n, len(imgs))
    if angles is None:
        angles = angle_fn(n, min_angle, max_angle, key)
    imgs = jnp.concatenate([imgs]*(len(angles)//len(imgs)+1), axis=0)
    imgs = imgs[:len(angles)]
    labels = jnp.concatenate([labels]*(len(angles)//len(labels)+1), axis=0)
    labels = labels[:len(angles)]
    imgs_rot = vmap(rotate_image)(imgs, angles)
    if include_labels:
        return imgs_rot, angles, labels
    
    return imgs_rot, angles


def load_target_digit_dataset(
    data_dir: str="/tmp/data",
    target_digit: int=0,
    dataset_type: str="fashion_mnist",
    n: int=None, 
):
    """Generate rotated dataset for a target digit.
    """
    dataset = load_base_dataset(data_dir, dataset_type, oh_train=False)
    train, val, test = \
        (_filter_target_digit(dataset[split], target_digit, n=n) 
         for split in ['train', 'val', 'test'])
    dataset = {
        'train': train,
        'val': val,
        'test': test,
    }
    
    return dataset


def load_rotated_dataset(
    dataset: dict=None,
    data_dir: str="/tmp/data",
    dataset_type: str="fashion_mnist",
    key: int=0,
    target_digit: int=None,
    angle_fn: Callable=None,
    min_angle: float=0.0,
    max_angle: float=180.0,
    ntrain: int=None,
    nval: int=None,
    ntest: int=None,
    include_labels: bool=True,
    match_train_test_angles: bool=False,
    angle_std = 5.0,
    **process_kwargs,
) -> dict:
    """Load rotated dataset.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if dataset is None:
        dataset = load_base_dataset(data_dir, dataset_type, oh_train=False)
    *keys, subkey = jr.split(key, 3)
    train, val = \
        (generate_rotated_images(*dataset[k], n, key_, target_digit, angle_fn,
                                min_angle, max_angle, include_labels)
        for k, n, key_ in zip(['train', 'val'], [ntrain, nval], keys))
    if match_train_test_angles:
        keys = jr.split(subkey, 2)
        train_angles = train[1]
        test_angles = train_angles + \
            jr.normal(keys[0], train_angles.shape) * angle_std
        test = generate_rotated_images(*dataset['test'], ntest, keys[1],
                                       target_digit, angle_fn, min_angle,
                                       max_angle, include_labels, test_angles)
    else:
        test = generate_rotated_images(*dataset['test'], ntest, subkey,
                                       target_digit, angle_fn, min_angle,
                                       max_angle, include_labels)
    oh_train = True if include_labels else False
    output_dim = train[2].max() + 1
    dataset = process_dataset(train, val, test, oh_train=oh_train, 
                              output_dim=output_dim, shuffle=False, 
                              **process_kwargs)
    
    return dataset
    
    
def load_seq_digits_rotated_dataset(
    n_tasks: int,
    ntrain_per_task: int,
    nval_per_task: int,
    ntest_per_task: int,
    data_dir: str="/tmp/data",
    dataset_type: str="fashion_mnist",
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
    for i in range(n_tasks):
        dataset = load_rotated_dataset(
            None, data_dir, dataset_type, key, i, angle_fn, min_angle, max_angle,
            ntrain=ntrain_per_task, nval=nval_per_task, ntest=ntest_per_task,
        )
        datasets.append(dataset)
    dataset = tree_map(
        lambda *args: jnp.concatenate(args, axis=0),
        *datasets,
    )
    
    return dataset


# Permuted Dataset -------------------------------------------------------------

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


def load_single_permuted_dataset(
    perm_idx: jnp.ndarray,
    ntrain: int,
    nval: int,
    ntest: int,
    dataset: dict=None,
    data_dir: str="/tmp/data",
    dataset_type: str="fashion_mnist",
    key: int=0,
    **process_kwargs,
) -> dict:
    """Load permuted dataset given single permuted index.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if dataset is None:
        dataset = load_base_dataset(data_dir, dataset_type, oh_train=False)
    output_dim = dataset['train'][1].max() + 1
    keys = jr.split(key, 3)
    
    train, val, test = \
        (_permute_dataset(*dataset[k], perm_idx, n, key_) for k, n, key_ in 
         zip(['train', 'val', 'test'], [ntrain, nval, ntest], keys))
    
    dataset = process_dataset(train, val, test, output_dim=output_dim,
                              **process_kwargs)
    
    return dataset


def load_permuted_dataset(
    n_tasks: int,
    ntrain_per_task: int,
    nval_per_task: int,
    ntest_per_task: int,
    dataset: dict=None,
    data_dir: str="/tmp/data",
    dataset_type: str="fashion_mnist",
    key: int=0,
    **process_kwargs,
) -> dict:
    """Load Permuted dataset.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if dataset is None:
        dataset = load_base_dataset(data_dir, dataset_type, oh_train=False)
    img_size = np.prod(dataset["train"][0][0].shape)
    identity_idx = jnp.arange(img_size)[None, :]
    keys = jr.split(key, n_tasks)
    perm_idx = jnp.concatenate(
        [identity_idx, jnp.array([jr.permutation(keys[i], jnp.arange(img_size))
                                  for i in range(n_tasks-1)]),],
        axis=0,
    )
    keys = jr.split(keys[-1], n_tasks)
    permute_fn = lambda idx, k: \
        load_single_permuted_dataset(idx, ntrain_per_task, nval_per_task,
                                     ntest_per_task, dataset, key=k,
                                     **process_kwargs)
    dataset = vmap(permute_fn)(perm_idx, keys)
    # Squash vmapped axis
    dataset = tree_map(lambda x: jnp.concatenate(x, axis=0), dataset)
    
    return dataset


# Rotated Permuted MNIST -------------------------------------------------------

def load_rotated_permuted_mnist_dataset(
    n_tasks: int,
    ntrain_per_task: int,
    nval_per_task: int,
    ntest_per_task: int,
    dataset: dict=None,
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
                                    ntest_per_task, dataset, data_dir, fashion, 
                                    keys[0], oh_train=False)
    dataset = load_rotated_mnist_dataset(
        permuted_dataset, angle_fn=angle_fn, min_angle=min_angle, 
        max_angle=max_angle, key=keys[1], include_labels=False,
    )
    
    return dataset


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
    dataset_type: str="fashion_mnist",
    key=0,
) -> dict:
    """Load Split MNIST dataset, each with 2 consecutive digits given dataset.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    dataset = load_base_dataset(data_dir, dataset_type, oh_train=False)
    train, val, test = \
        (make_split_mnist(*dataset[k], n)
         for k, n in zip(('train', 'val', 'test'),
                         (ntrain_per_task, nval_per_task, ntest_per_task)))
    dataset = process_mnist_dataset(train, val, test, shuffle=False,
                                    oh_train=False)
    
    return dataset


# For experiments --------------------------------------------------------------

def generate_mnist_experiment(
    ntrain: int
):
    kwargs = {
        "ntrain": ntrain,
        "nval": 1_000,
    }
    dataset = {
        "load_fn": partial(load_base_dataset, **kwargs),
        "configs": kwargs,
    }
    
    return dataset


def generate_pmnist_experiment(
    n_tasks: int=10,
    ntrain_per_task: int=300,
    nval_per_task: int=1,
    ntest_per_task: int=500,
):
    kwargs = {
        "n_tasks": n_tasks,
        "ntrain_per_task": ntrain_per_task,
        "nval_per_task": nval_per_task,
        "ntest_per_task": ntest_per_task,
    }
    dataset = {
        "load_fn": partial(load_permuted_mnist_dataset, **kwargs),
        "configs": kwargs,
    }
    
    return dataset


def generate_rmnist_experiment(
    ntrain: int,
    angle_fn: Callable,
    include_labels: bool=False,
    max_angle: float=180.0
):
    kwargs = {
        "ntrain": ntrain,
        "nval": 500,
    }
    load_fn = partial(load_rotated_mnist_dataset, include_labels=include_labels,
                      angle_fn=angle_fn, max_angle=max_angle, **kwargs)
    if angle_fn in (generate_amplified_angles, generate_random_walk_angles):
        load_fn = partial(load_fn, match_train_test_angles=True)
    dataset = {
        "load_fn": load_fn,
        "configs": kwargs,
    }
    
    return dataset


def generate_split_mnist_experiment(
    ntrain_per_task: int=300,
    nval_per_task: int=1,
    ntest_per_task: int=500,
):
    kwargs = {
        "ntrain_per_task": ntrain_per_task,
        "nval_per_task": nval_per_task,
        "ntest_per_task": ntest_per_task,
    }
    dataset = {
        "load_fn": partial(load_split_mnist_dataset, **kwargs),
        "configs": kwargs,
    }
    
    return dataset


def generate_rotated_permuted_mnist_experiment(
    n_tasks: int=10,
    ntrain_per_task: int=300,
    nval_per_task: int=1,
    ntest_per_task: int=500,
    angle_fn: Callable=None,
    min_angle: float=0.0,
    max_angle: float=180.0,
):
    kwargs = {
        "n_tasks": n_tasks,
        "ntrain_per_task": ntrain_per_task,
        "nval_per_task": nval_per_task,
        "ntest_per_task": ntest_per_task,
        "angle_fn": angle_fn,
        "min_angle": min_angle,
        "max_angle": max_angle,
    }
    dataset = {
        "load_fn": partial(load_rotated_permuted_mnist_dataset, **kwargs),
        "configs": kwargs,
    }
    
    return dataset


smnist_kwargs = {
    'ntrain_per_task': 300,
    'nval_per_task': 1,
    'ntest_per_task': 500,
}


clf_datasets = {
    'stationary-mnist': generate_mnist_experiment,
    'permuted-mnist': generate_pmnist_experiment,
    'rotated-mnist': partial(generate_rmnist_experiment,
                             angle_fn=generate_random_walk_angles,
                             include_labels=True, max_angle=90.0),
    'split-mnist': generate_split_mnist_experiment,
}


reg_datasets = {
    "iid-mnist": partial(generate_rmnist_experiment,
                         angle_fn=generate_random_angles),
    "amplified-mnist": partial(generate_rmnist_experiment,
                               angle_fn=generate_amplified_angles), 
    "random-walk-mnist": partial(generate_rmnist_experiment,
                                 angle_fn=generate_random_walk_angles),
    "permuted-mnist": partial(generate_rotated_permuted_mnist_experiment,
                              angle_fn=generate_random_angles),
}
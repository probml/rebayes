from functools import partial
from typing import Callable, Tuple, Union

import augmax
from augmax.geometric import GeometricTransformation, LazyCoordinates
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
        (_process_mnist(dataset, key, shuffle, oh)
         for dataset, key, oh in zip([train, val, test], keys,
                                     [oh_train, False, False]))
    dataset = {
        'train': train,
        'val': val,
        'test': test,
    }
    
    return dataset




# MNIST ------------------------------------------------------------------------

# def load_mnist_dataset(
#     data_dir: str="/tmp/data",
#     fashion: bool=False,
#     n_train: int=None,
#     n_val: int=None,
#     n_test: int=None,
#     oh_train: int=True,
# ) -> dict:
#     """Load MNIST train, validatoin, and test datasets into memory.
#     """
#     dataset='mnist'
#     if fashion:
#         dataset='fashion_mnist'
#     ds_builder = tfds.builder(dataset, data_dir=data_dir)
#     ds_builder.download_and_prepare()
    
#     train_ds, val_ds, test_ds = \
#         (tfds.as_numpy(ds_builder.as_dataset(split=split, batch_size=-1)) 
#          for split in ['train[10%:]', 'train[:10%]', 'test'])
    
#     # Normalize pixel values
#     for ds in [train_ds, val_ds, test_ds]:
#         ds['image'] = np.float32(ds['image']) / 255.
    
#     # If specified, only return a subset of the data
#     n_train, n_val, n_test = \
#         (min(n, len(ds['image'])) if n else len(ds['image'])
#         for n, ds in zip([n_train, n_val, n_test], [train_ds, val_ds, test_ds]))
    
#     train = [train_ds[key][:n_train] for key in ['image', 'label']]
#     val = [val_ds[key][:n_val] for key in ['image', 'label']]
#     test = [test_ds[key][:n_test] for key in ['image', 'label']]
    
#     dataset = process_mnist_dataset(train, val, test, shuffle=True, 
#                                     oh_train=oh_train)

#     return dataset

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


# def rotate_mnist_dataset(X, angles):
#     X_rotated = vmap(rotate_mnist)(X, angles)
    
#     return X_rotated


# def generate_rotating_mnist_dataset(X=None, min_angle=0, max_angle=180, key=0, 
#                                     target_digit=None, generate_angle_fn=None,
#                                     n_per_task=None):
#     if isinstance(key, int):
#         key = jr.PRNGKey(key)
#     if X is None:
#         mnist_dataset = load_mnist_dataset()
#         X, y = mnist_dataset["train"]
#         if target_digit is not None:
#             X = X[y == target_digit]
#         if n_per_task is not None:
#             X = X[:n_per_task]
#     if generate_angle_fn is None:
#         generate_angle_fn = generate_random_angles
#     y = generate_angle_fn(len(X), min_angle, max_angle, key)
#     X_rotated = rotate_mnist_dataset(X, y)
#     y_mean, y_std = y.mean(), y.std()
#     y_normalized = (y - y_mean) / y_std
    
#     return X_rotated, y_normalized, y_mean, y_std

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
    
    return imgs_rot, angles, labels


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
                                 min_angle, max_angle) 
         for k, n, key_ in zip(['train', 'val', 'test'], [ntrain, nval, ntest],
                               jr.split(key, 3)))
    dataset = process_mnist_dataset(train, val, test, **process_kwargs)
    
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
    perm_idx = jnp.concatenate(
        [identity_idx, jnp.array([jr.permutation(key, jnp.arange(28*28))
                                  for _ in range(n_tasks-1)]),],
        axis=0,
    )
    keys = jr.split(key, n_tasks)
    permute_fn = lambda idx, k: \
        load_single_permuted_mnist_dataset(idx, ntrain_per_task, nval_per_task,
                                           ntest_per_task, dataset, key=k,
                                           **process_kwargs)
    dataset = vmap(permute_fn)(perm_idx, keys)
    # Squash vmapped axis
    dataset = tree_map(lambda x: jnp.concatenate(x, axis=0), dataset)
    
    return dataset

# def load_permuted_mnist_dataset(
#     n_tasks: int,
#     ntrain_per_task: int,
#     nval_per_task: int,
#     ntest_per_task: int,
#     data_dir: str="/tmp/data",
#     fashion: bool=False,
#     key: int=0,
# ) -> dict:
#     """Load permuted MNIST dataset.
#     """
#     if isinstance(key, int):
#         key = jr.PRNGKey(key)   
#     dataset = load_mnist_dataset(data_dir, fashion, oh_train=True)
#     # n_per_task = {
#     #     'train': ntrain_per_task,
#     #     'val': nval_per_task,
#     #     'test': ntest_per_task
#     # }
#     # result = {data_type: ([], []) for data_type in ['train', 'val', 'test']}
#     perm_idxs = [jr.permutation(key, jnp.arange(28*28)) for _ in range(n_tasks)]
    
#     for i in range(n_tasks):
#         key, subkey = jr.split(key)
#         perm_idx = jr.permutation(subkey, jnp.arange(28*28))
#         if i == 0:
#             perm_idx = jnp.arange(28*28) # Identity permutation for first task
#         permute_fn = partial(_permute, idx=perm_idx)
        
#         for data_type, data in dataset.items():
#             key, subkey = jr.split(key)
#             X, Y = data
#             sample_idx = jr.choice(subkey, jnp.arange(len(X)), shape=(n_per_task[data_type],), replace=False)
            
#             curr_X = vmap(permute_fn)(X[sample_idx])
#             result[data_type][0].append(curr_X)
            
#             curr_Y = Y[sample_idx]
#             result[data_type][1].append(curr_Y)
    
#     for data_type in ['train', 'val', 'test']:
#         result[data_type] = (jnp.concatenate(result[data_type][0]), jnp.concatenate(result[data_type][1]))
            
#     return result


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


# def generate_rotating_permuted_mnist_regression_dataset(
#     n_tasks, ntrain_per_task, nval_per_task, ntest_per_task, min_angle=0, 
#     max_angle=180, key=0, fashion=True, mnist_dataset = None
# ):
#     if mnist_dataset is None:
#         mnist_dataset = load_permuted_mnist_dataset(n_tasks, ntrain_per_task, 
#                                                     nval_per_task, ntest_per_task, 
#                                                     key, fashion)
#     dataset = {
#         k: generate_rotating_mnist_dataset(mnist_dataset[k][0], min_angle, max_angle, i)
#         for i, k in enumerate(('train', 'val', 'test'))
#     }
    
#     return dataset


# Split MNIST ------------------------------------------------------------------

# def load_split_mnist_dataset(ntrain_per_task, nval_per_task, ntest_per_task, key=0, fashion=False):
#     if isinstance(key, int):
#         key = jr.PRNGKey(key)
    
#     smnist_kwargs = {
#         "class_ids_from_zero_in_each_exp": True,
#         "fixed_class_order": range(10),
#     }
#     if fashion:
#         dataloader = SplitFMNIST
#     else:
#         dataloader = SplitMNIST
#     dataset = load_avalanche_mnist_dataset(dataloader, 5, ntrain_per_task, 
#                                            ntrain_per_task, nval_per_task, 
#                                            ntest_per_task, key=key, oh_train=False,
#                                            **smnist_kwargs)
    
#     return dataset

# def make_split_mnist(fashion=False):
#     mnist_dataset = load_mnist_dataset(fashion=fashion)
#     train, test = mnist_dataset['train'], mnist_dataset['test']
#     images, labels = train
#     images_test, labels_test = test
#     X_train, y_train = jnp.array(images), jnp.array(labels.ravel())
#     X_test, y_test = jnp.array(images_test), jnp.array(labels_test.ravel())

#     train_set_by_digit_pair, test_set_by_digit_pair = {}, {}
#     for i in range(10//2):
#         train_indx = (y_train == 2*i) | (y_train == 2*i+1)
#         train_set_by_digit_pair[str(2*i)+str(2*i+1)] = (X_train[train_indx], y_train[train_indx]-2*i, y_train[train_indx])
#         test_indx = (y_test == 2*i) | (y_test == 2*i+1)
#         test_set_by_digit_pair[str(2*i)+str(2*i+1)] = (X_test[test_indx], y_test[test_indx]-2*i, y_test[test_indx])
#     return train_set_by_digit_pair, test_set_by_digit_pair

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


# def get_data_for_tasks(tasks, data_set_per_digit_pair, ndata_per_dist):
#     Xs, Ys, Ls, Ids = [], [],  [], []
#     for i in tasks:
#         name = str(2*i)+str(2*i+1)
#         Xi, Yi, Li = data_set_per_digit_pair[name]
#         src_ndx = jnp.arange(ndata_per_dist)
#         Xs.append(Xi[src_ndx])
#         Ys.append(Yi[src_ndx])
#         Ls.append(Li[src_ndx])
#         identifiers = int(i) * jnp.ones(len(src_ndx), dtype=int)
#         Ids.append(identifiers)
#     X = jnp.concatenate(Xs)
#     Y = jnp.concatenate(Ys)
#     L = jnp.concatenate(Ls)
#     Id = jnp.concatenate(Ids)
#     return X, Y, L, Id


# class SplitMNIST():
#     def __init__(
#             self,
#             ntrain_per_task: int,
#             ntest_per_task: int
#     ):
#         self.ntrain_per_task = ntrain_per_task
#         self.ntest_per_task = ntest_per_task
#         (self.train_set_by_digit_pair, self.test_set_by_digit_pair) = make_split_mnist()

#     def get_training_data_all_tasks(self):
#         ntasks = 5
#         Xtr, Ytr, Ltr, Itr = get_data_for_tasks(np.arange(ntasks), self.train_set_by_digit_pair, self.ntrain_per_task)
#         return Xtr, Ytr, Ltr, Itr
    
#     def get_test_data_for_task(self, task):
#         Xte, Yte, Ltr, Ite = get_data_for_tasks([task], self.test_set_by_digit_pair, self.ntest_per_task)
#         return Xte, Yte, Ltr, Ite
    
#     def get_test_data_for_seen_tasks(self, task):
#         Xte, Yte, Lte, Ite = get_data_for_tasks(jnp.arange(task+1), self.test_set_by_digit_pair, self.ntest_per_task)
#         return Xte, Yte, Lte, Ite
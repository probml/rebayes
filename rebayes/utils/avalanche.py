from pathlib import Path
from typing import Optional, Any, Union

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import DataLoader
import jax_dataloader.core as jdl
from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.classic.cmnist import (
    PixelsPermutation,
)
from avalanche.benchmarks.classic.cfashion_mnist import (
    _default_fmnist_train_transform,
    _default_fmnist_eval_transform,
)
from avalanche.benchmarks.datasets.external_datasets.fmnist import get_fmnist_dataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.transform_groups import DefaultTransformGroups
from avalanche.benchmarks.utils.classification_dataset import make_classification_dataset



def flatten_batch(X):
    if type(X)==torch.Tensor: X = jnp.array(X.numpy())
    if type(X)==np.ndarray: X = jnp.array(X)
    sz = jnp.array(list(X.shape))
    batch_size  = sz[0]
    other_size = jnp.prod(sz[1:])
    X = X.flatten().reshape(batch_size, other_size)
    return X


def dataloader_to_numpy(dataloader):
    # data = np.array(train_dataloader.dataset) # mangles the shapes
    all_X = []
    all_y = []
    for X, y in dataloader:
        all_X.append(X)
        all_y.append(y)
    X = torch.cat(all_X, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    if y.ndim == 1:
        y = y[:, None]
    return X, y


def avalanche_dataloader_to_numpy(dataloader):
    # data = np.array(train_dataloader.dataset) # mangles the shapes
    all_X = []
    all_y = []
    for X, y, t in dataloader: # avalanche uses x,y,t
        all_X.append(X)
        all_y.append(y)
    X = torch.cat(all_X, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    if y.ndim == 1:
        y = y[:, None]
    return X, y


def make_avalanche_datasets_pytorch(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch, key):
    '''Make pytorch dataloaders from avalanche dataset.
    ntrain_per_dist: number of training examples from each distribution (experience).
    batch_size: how many training examples per batch.
    ntest_per_batch: how many test examples per training batch.
    '''
    train_stream = dataset.train_stream
    test_stream = dataset.test_stream
    nexperiences = len(train_stream) # num. distinct distributions
    nbatches_per_dist = int(ntrain_per_dist / ntrain_per_batch)
    ntest_per_dist = ntest_per_batch * nbatches_per_dist

    train_sets = []
    test_sets = []
    for exp in trange(nexperiences, desc='Translating avalanche dataset to pytorch...'):
        key, *subkeys = jr.split(key, 3)
        ds = train_stream[exp].dataset
        train_ndx = jr.choice(subkeys[0], len(ds), shape=(ntrain_per_dist,), replace=False)
        train_set = torch.utils.data.Subset(ds, train_ndx)
        train_sets.append(train_set)

        ds = test_stream[exp].dataset
        test_ndx = jr.choice(subkeys[1], len(ds), shape=(ntest_per_dist,), replace=False)
        test_set = torch.utils.data.Subset(ds, test_ndx)
        test_sets.append(test_set)

    train_set = torch.utils.data.ConcatDataset(train_sets)
    test_set = torch.utils.data.ConcatDataset(test_sets)
    
    return train_set, test_set


def make_avalanche_data(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    train_set, test_set = make_avalanche_datasets_pytorch(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch, key)
    train_dataloader = DataLoader(train_set, batch_size=ntrain_per_batch, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=ntest_per_batch, shuffle=False)
    Xtr, Ytr = avalanche_dataloader_to_numpy(train_dataloader)
    Xte, Yte = avalanche_dataloader_to_numpy(test_dataloader)
    return  jnp.array(Xtr), jnp.array(Ytr), jnp.array(Xte), jnp.array(Yte)


def make_avalanche_dataloaders_numpy(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch):
    Xtr, Ytr, Xte, Yte = make_avalanche_data(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch)
    train_ds = jdl.Dataset(Xtr, Ytr)
    train_loader = jdl.DataLoaderJax(train_ds, batch_size=ntrain_per_batch, shuffle=False, drop_last=False)
    test_ds = jdl.Dataset(Xte, Yte)
    test_loader = jdl.DataLoaderJax(test_ds, batch_size=ntest_per_batch, shuffle=False, drop_last=False)
    return train_loader, test_loader


def PermutedFashionMNIST(
    n_experiences: int,
    *,
    return_task_id=False,
    seed: Optional[int] = None,
    train_transform: Optional[Any] = _default_fmnist_train_transform,
    eval_transform: Optional[Any] = _default_fmnist_eval_transform,
    dataset_root: Union[str, Path] = None
) -> NCScenario:
    """Modified from avalanche.benchmarks.classic.cmnist.PermutedMNIST"""
    list_train_dataset = []
    list_test_dataset = []
    rng_permute = np.random.RandomState(seed)

    mnist_train, mnist_test = get_fmnist_dataset(dataset_root)

    # for every incremental experience
    for _ in trange(n_experiences, desc="Generating PermutedFashionMNIST..."):
        # choose a random permutation of the pixels in the image
        idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(
            torch.int64
        )

        permutation = PixelsPermutation(idx_permute)

        # Freeze the permutation
        permuted_train = make_avalanche_dataset(
            make_classification_dataset(mnist_train),
            frozen_transform_groups=DefaultTransformGroups((permutation, None)),
        )

        permuted_test = make_avalanche_dataset(
            make_classification_dataset(mnist_test),
            frozen_transform_groups=DefaultTransformGroups((permutation, None)),
        )

        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)

    return nc_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=return_task_id,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
from typing import Sequence
from functools import partial
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import flax.linen as nn
from jax.flatten_util import ravel_pytree
from jax.experimental import host_callback
from jax import jacrev
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
import jax_dataloader.core as jdl


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
    for exp in range(nexperiences):
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



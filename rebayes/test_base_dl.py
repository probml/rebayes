
#pytest test_base_dl.py  -rP
# Test the dataloader version

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import time
from functools import partial

from jax import jit
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
import chex

import haiku as hk


import jax_dataloader.core as jdl

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T

from rebayes.base import RebayesParams, Rebayes, Belief, make_rebayes_params


class RebayesSum(Rebayes):
    """The belief state is the sum of all the input X_t values."""
    def __init__(
        self,
        params: RebayesParams,
        shape_in, 
        shape_out
    ):
        self.params = params
        self.shape_in = shape_in
        self.shape_out = shape_out

    def init_bel(self) -> Belief:
        bel = Belief(dummy = jnp.zeros(self.shape_in))
        return bel
    
    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        Y: Float[Array, "obs_dim"]
    ) -> Belief:
        return Belief(dummy = bel.dummy + X)

    
    
def callback_dl(b, bel_pre, bel, batch):
    jax.debug.print("callback on batch {b}", b=b)
    Xtr, Ytr = batch
    jax.debug.print("Xtr shape {x1}, Ytr shape {y1}", x1=Xtr.shape, y1=Ytr.shape)
    return b


def make_data():
    keys = hk.PRNGSequence(42)
    ndim_in = 5
    nclasses = 10
    ntime = 12
    #X = jnp.arange(ntime).reshape((ntime, 1)) # 1d
    X = jr.normal(next(keys), (ntime, ndim_in))
    labels = jr.randint(next(keys), (ntime,), 0,  nclasses-1)
    Y = jax.nn.one_hot(labels, nclasses)
    return X, Y

def test_scan_dataloader():
    print('test scan dataloaders')
    Xtr, Ytr = make_data()
    train_ds = jdl.Dataset(Xtr, Ytr)
    train_loader = jdl.DataLoaderJax(train_ds, batch_size=5, shuffle=False, drop_last=False)
    shape_in = Xtr.shape[1:]
    shape_out = Ytr.shape[1:]
    estimator = RebayesSum(make_rebayes_params(), shape_in, shape_out)
    bel, outputs = estimator.scan_dataloader(train_loader, callback=callback_dl)
    Xsum = jnp.sum(Xtr, axis=0)
    assert(jnp.allclose(Xsum, bel.dummy, atol=1e-2))
   
def test_scan_dataloader_batch1():
    print('test scan dataloader batch1')
    # when batchsize=1, scan_dataloader == scan 
    Xtr, Ytr = make_data()
    train_ds = jdl.Dataset(Xtr, Ytr)
    train_loader = jdl.DataLoaderJax(train_ds, batch_size=1, shuffle=False, drop_last=False)
    shape_in = Xtr.shape[1:]
    shape_out = Ytr.shape[1:]
    estimator = RebayesSum(make_rebayes_params(), shape_in, shape_out)
    bel, outputs = estimator.scan_dataloader(train_loader)
    bel2, outputs2 = estimator.scan(Xtr, Ytr)
    assert(jnp.allclose(bel.dummy, bel2.dummy))

def make_mnist_data():
    # convert PIL to pytorch tensor, flatten (1,28,28) to (784), standardize values
    # using mean and std deviation of the MNIST dataset.
    transform=T.Compose([T.ToTensor(),
                        T.Normalize((0.1307,), (0.3081,)),
                        T.Lambda(torch.flatten)]
                        ) 

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # convert full dataset to numpy
    Xtr, Ytr = train_set.data.numpy(), train_set.targets.numpy()
    Xte, Yte = test_set.data.numpy(), test_set.targets.numpy()

    # extract small subset
    ntrain, ntest = 100, 500
    train_ndx, test_ndx = jnp.arange(ntrain), jnp.arange(ntest)

    return Xtr[train_ndx], Ytr[train_ndx], Xte[test_ndx], Yte[test_ndx]


def test_mnist():
    print('test mnist')
    Xtr, Ytr, Xte, Yte = make_mnist_data()
    train_ds = jdl.Dataset(Xtr, Ytr)
    train_loader = jdl.DataLoaderJax(train_ds, batch_size=50, shuffle=False, drop_last=False)
    shape_in = Xtr.shape[1:]
    shape_out = Ytr.shape[1:]
    estimator = RebayesSum(make_rebayes_params(), shape_in, shape_out)
    bel, outputs = estimator.scan_dataloader(train_loader, callback=callback_dl)
    Xsum = jnp.sum(Xtr, axis=0)
    assert(jnp.allclose(Xsum, bel.dummy, atol=1e-2))
    


#pytest test_base.py  -rP

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
        ndim_in: int, 
        ndim_out: int
    ):
        self.params = params
        self.ndim_in = ndim_in
        self.ndim_out = ndim_out

    def init_bel(self) -> Belief:
        bel = Belief(dummy = jnp.zeros((self.ndim_in,)))
        return bel
    
    def update_state(
        self,
        bel: Belief,
        X: Float[Array, "input_dim"],
        Y: Float[Array, "obs_dim"]
    ) -> Belief:
        return Belief(dummy = bel.dummy + X)

    
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


def callback_scan(bel, pred_obs, t, X, Y, **kwargs):
    jax.debug.print("callback with t={t}", t=t)
    return t

def test_scan():
    print('test scan')
    X, Y = make_data()
    ndim_in = X.shape[1]
    ndim_out = Y.shape[1]
    estimator = RebayesSum(make_rebayes_params(), ndim_in, ndim_out)
    bel, outputs = estimator.scan(X, Y, callback=callback_scan, progress_bar=False)
    print('final belief ', bel)
    print('outputs ', outputs)
    Xsum = jnp.sum(X, axis=0)
    assert jnp.allclose(bel.dummy, Xsum)

def test_update_batch():
    print('test update batch')
    X, Y = make_data()
    ndim_in = X.shape[1]
    ndim_out = Y.shape[1]
    estimator = RebayesSum(make_rebayes_params(), ndim_in, ndim_out)
    Xsum = jnp.sum(X, axis=0)

    bel = estimator.init_bel()
    bel = estimator.update_state_batch(bel, X, Y)
    assert(jnp.allclose(bel.dummy, Xsum))

    bel = estimator.init_bel()
    N = X.shape[0]
    for n in range(N):
        bel = estimator.update_state(bel, X[n], Y[n])
    assert(jnp.allclose(bel.dummy, Xsum))


    
def callback_dl(bel, bel_pre_update, b, Xtr, Ytr, Xte, Yte, **kwargs):
    jax.debug.print("callback on batch {b}", b=b)
    jax.debug.print("Xtr shape {x1}, Ytr shape {y1}", x1=Xtr.shape, y1=Ytr.shape)
    jax.debug.print("Xte shape {x1}, Yte shape {y1}", x1=Xte.shape, y1=Yte.shape)
    #jax.debug.print("pre dummy {a}, post dummy {b}", a=bel_pre_update.dummy, b=bel.dummy)
    return b

def make_dataloaders(X, Y, batch_size=5):
    # convert from jax to numpy to pytorch
    train_ds = torch.utils.data.TensorDataset(torch.Tensor(np.array(X)), torch.Tensor(np.array(Y)))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(train_ds, batch_size=X.shape[0], shuffle=False) # whole dataset
    return train_loader, test_loader

def test_scan_dataloaders():
    print('test scan dataloaders')
    X, Y = make_data()
    ndim_in = X.shape[1]
    ndim_out = Y.shape[1]
    estimator = RebayesSum(make_rebayes_params(), ndim_in, ndim_out)
    train_loader, test_loader = make_dataloaders(X, Y)
    bel, outputs = estimator.scan_dataloader(train_loader, test_loader, callback_dl, verbose=True)
    Xsum = jnp.sum(X,axis=0)
    print(bel, Xsum)
    assert jnp.allclose(bel.dummy, Xsum)

def dataloader_to_numpy(dataloader):
  # is there a faster way?
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
    #Xtr, Ytr = train_set.data.numpy(), train_set.targets.numpy()
    #Xte, Yte = test_set.data.numpy(), test_set.targets.numpy()
    #print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape) # not transformed!!

    # extract small subset
    ntrain, ntest = 100, 500
    train_ndx, test_ndx = range(ntrain), range(ntest)
    train_set = torch.utils.data.Subset(train_set, train_ndx)
    test_set = torch.utils.data.Subset(test_set, test_ndx)

    train_dataloader = DataLoader(train_set, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=False) # single big batch

    Xtr, Ytr = dataloader_to_numpy(train_dataloader)
    Xte, Yte = dataloader_to_numpy(test_dataloader)

    return train_dataloader, test_dataloader, Xtr, Ytr, Xte, Yte

def test_mnist():
    print('test mnist')
    train_dataloader, test_dataloader, Xtr, Ytr, Xte, Yte = make_mnist_data()
    print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)
    ndim_in = Xtr.shape[1]
    ndim_out = Ytr.shape[1]
    estimator = RebayesSum(make_rebayes_params(), ndim_in, ndim_out)
    bel, outputs = estimator.scan_dataloader(train_dataloader, test_dataloader, callback_dl, verbose=True)
    Xsum = jnp.sum(Xtr, axis=0)
    #print(bel.dummy[100:110])
    #print(Xsum[100:110])
    assert(jnp.allclose(Xsum, bel.dummy, atol=1e-2))

# https://github.com/BirkhoffG/jax-dataloader

#pytest jdl_test.py  -rP

import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr

import jax_dataloader.core as jdl

from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

def test_mnist():
    train_ds = MNIST('/tmp/mnist/', download=True, transform=ToTensor())
    X, Y = train_ds.data.numpy(), train_ds.targets.numpy()
    ds = jdl.Dataset(X, Y)
    train_loader = jdl.DataLoaderJax(ds, batch_size=5, shuffle=False, drop_last=False)
    train_iter = iter(train_loader)
    Xtr, Ytr = next(train_iter)
    assert(Xtr.shape == (5,28,28))
    assert(Ytr.shape == (5,))


def make_data():
    keys = hk.PRNGSequence(42)
    ndim_in = 2
    nclasses = 10
    ndata = 12
    X = jr.normal(next(keys), (ndata, ndim_in))
    labels = jr.randint(next(keys), (ndata,), 0,  nclasses-1)
    Y = jax.nn.one_hot(labels, nclasses)
    return X, Y, labels

def test_tabular():
    X, Y, labels = make_data()
    print('batch data shape', X.shape, Y.shape, labels.shape)
    print('labels ', labels)

    ds = jdl.Dataset(labels)
    train_loader = jdl.DataLoaderJax(ds, batch_size=5, shuffle=False, drop_last=False)
    for i, Ltr in enumerate(train_loader):
        print('batch ' ,i, Ltr)

    ds = jdl.Dataset(X,Y)
    train_loader = jdl.DataLoaderJax(ds, batch_size=5, shuffle=False, drop_last=False)
    train_iter = iter(train_loader)
    for b in range(len(train_iter)):
        Xtr, Ytr = next(train_iter)
        print('batch ', b, Xtr.shape, Ytr.shape)
        if b < 2:
            assert(Xtr.shape == (5, 2))
        else:
            assert(Xtr.shape == (2,2))

def main():
    test_mnist()
    test_tabular()

if __name__ == "__main__":
    main()

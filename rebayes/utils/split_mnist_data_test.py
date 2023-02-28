

#pytest split_mnist_data_test.py  -rP

import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any

import jax_dataloader.core as jdl

from rebayes.utils.split_mnist_data import SplitMNIST
from rebayes.base import Rebayes, RebayesParams, make_rebayes_params, Belief

def test_split_mnist():
    ntrain_per_task = 10
    ntest_per_task = 4
    split_mnist = SplitMNIST(ntrain_per_task, ntest_per_task)
    X_01, y_01, L_01 = split_mnist.test_set_by_digit_pair['01']
    X_23, y_23, L_23 = split_mnist.test_set_by_digit_pair['23']
    print(X_01.shape, X_23.shape)
    assert all(np.unique(L_23)==np.array([2, 3]))
    assert all(np.unique(y_23)==np.array([0, 1]))

    X, Y, L, I = split_mnist.get_test_data_for_task(1)
    X_23, y_23, L_23 = split_mnist.test_set_by_digit_pair['23']
    print(X.shape, X_23.shape)
    jnp.allclose(X, X_23[:ntest_per_task])
    print(Y.shape, y_23.shape)
    jnp.allclose(Y, y_23[:ntest_per_task])

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
    
def test_split_mnist_rebayes():
    ntrain_per_task = 10
    ntest_per_task = 4
    split_mnist = SplitMNIST(ntrain_per_task, ntest_per_task)

    def callback_dl(b, bel_pre, bel_post, batch):
        jax.debug.print("callback on batch {b}", b=b)
        Xtr, Ytr, Ltr, Itr = batch
        jax.debug.print("Xtr shape {x1}, Ytr shape {y1}", x1=Xtr.shape, y1=Ytr.shape)
        task = int(Itr[0])
        Xte, Yte, Lte, Ite = split_mnist.get_test_data_for_seen_tasks(task)
        jax.debug.print("Xte shape {x1}, Yte shape {y1}", x1=Xte.shape, y1=Yte.shape)
        print('train labels ', Ltr)
        print('test labels ', Lte)
        #plot_batch(Xte, Yte, ttl='batch {:d}'.format(b))
        return (Ltr, Itr, Lte, Ite)

    Xtr, Ytr, Ltr, Itr = split_mnist.get_training_data_all_tasks() 
    print(Xtr.shape, Ytr.shape, Ltr.shape, Itr.shape)

    train_ds = jdl.Dataset(Xtr, Ytr, Ltr, Itr)
    ntrain_per_batch = 5
    train_loader = jdl.DataLoaderJax(train_ds, batch_size=ntrain_per_batch, shuffle=False, drop_last=False)

    shape_in = Xtr.shape[1:]
    shape_out = 1
    estimator = RebayesSum(make_rebayes_params(), shape_in, shape_out)

    bel, outputs = estimator.scan_dataloader(train_loader, callback=callback_dl)
    Xsum = jnp.sum(Xtr, axis=0)
    assert(jnp.allclose(Xsum, bel.dummy, atol=1e-2))

    for b in range(len(train_loader)):
        print('batch ', b)
        out = outputs[b]
        print(out)
        (Ltr, Itr, Lte, Ite) = out
        task = Itr[0]
        Xte, Yte, Lte_expected, Ite_expected = split_mnist.get_test_data_for_seen_tasks(task)
        assert jnp.allclose(Lte_expected, Lte)

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple
from jaxtyping import Float, Int, Array
from flax.training.train_state import TrainState

@partial(jax.jit, static_argnames=("applyfn",))
def lossfn(params, X, y, applyfn):
    yhat = applyfn(params, X)
    mll = (y - yhat.ravel()) ** 2
    return mll.mean()


@partial(jax.jit, static_argnames=("applyfn",))
def rmae(params, X, y, applyfn):
    yhat = applyfn(params, X)
    err = jnp.abs(y - yhat.ravel())
    return err.mean()


@partial(jax.jit, static_argnames=("loss_grad",))
def train_step(
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    ixs: Int[Array, "batch_len"],
    state: TrainState,
    loss_grad: Callable,
) -> Tuple[float, TrainState]:
    """
    Perform a single training step.
    The `loss_grad` function
    """
    X_batch = X[ixs]
    y_batch = y[ixs]
    loss, grads = loss_grad(state.params, X_batch, y_batch, state.apply_fn)    
    state = state.apply_gradients(grads=grads)
    return loss, state


@partial(jax.jit, static_argnums=(1,2))
def get_batch_train_ixs(key, num_samples, batch_size):
    """
    Obtain the training indices to be used in an epoch of
    mini-batch optimisation.
    """
    steps_per_epoch = num_samples // batch_size
    
    batch_ixs = jax.random.permutation(key, num_samples)
    batch_ixs = batch_ixs[:steps_per_epoch * batch_size]
    batch_ixs = batch_ixs.reshape(steps_per_epoch, batch_size)
    
    return batch_ixs


def train_epoch(
    key: int,
    batch_size: int,
    state: TrainState,
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    loss_grad: Callable,
):
    num_train = X.shape[0]
    loss_epoch = 0.0
    train_ixs = get_batch_train_ixs(key, num_train, batch_size)
    for ixs in train_ixs:
        loss, state = train_step(X, y, ixs, state, loss_grad)
        loss_epoch += loss
    return loss_epoch, state


@partial(jax.jit, static_argnames=("buffer_size",))
def get_fifo_batches(ix, buffer_size):
    ix_lookback = (ix - buffer_size) + 1
    batches = jnp.arange(buffer_size) + ix_lookback
    return batches


@partial(jax.jit, static_argnames=("applyfn",))
def lossfn_fifo(params, X, y, ixs, applyfn):
    X_batch, y_batch = X[ixs], y[ixs]
    counter = (ixs >= 0)
    
    yhat = applyfn(params, X_batch).ravel()
    loss = (y_batch - yhat) ** 2
    loss = (loss * counter).sum() / counter.sum()
    return loss
    

@partial(jax.jit, static_argnames=("loss_grad",))
def train_step_fifo(
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs dim_obs"],
    ixs: Int[Array, "batch_len"],
    state: TrainState,
    loss_grad: Callable,
) -> Tuple[float, TrainState]:
    """
    """
    loss, grads = loss_grad(state.params, X, y, ixs, state.apply_fn)
    state = state.apply_gradients(grads=grads)
    return loss, state
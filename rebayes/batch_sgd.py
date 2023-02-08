import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple, Union
from jaxtyping import Float, Int, Array, PyTree
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


def train_full(
    key: jax.random.PRNGKey,
    num_epochs: int,
    batch_size: int,
    state: TrainState,
    X: Float[Array, "num_obs dim_obs"],
    y: Float[Array, "num_obs"],
    loss: Callable[[PyTree, Float[Array, "num_obs dim_obs"], Float[Array, "num_obs"], Callable], float],
    X_test: Union[None, Float[Array, "num_obs_test dim_obs"]] = None,
    y_test: Union[None, Float[Array, "num_obs_test"]] = None,
):
    loss_grad = jax.value_and_grad(loss, 0)

    def epoch_step(state, t):
        keyt = jax.random.fold_in(key, t)
        loss_train, state = train_epoch(keyt, batch_size, state, X, y, loss_grad)

        if (X_test is not None) and (y_test is not None):
            loss_test = loss(state.params, X_test, y_test, state.apply_fn)
        else:
            loss_test = None

        losses = {
            "train": loss_train,
            "test": loss_test,
        } 
        return state, losses
    steps = jnp.arange(num_epochs)
    state, losses = jax.lax.scan(epoch_step, state, steps)
    return state, losses


class ReFifoSGD:
    """
    Recursive FIFO SGD.
    """
    def __init__(self, buffer_size, lossfn, n_inner=1):
        self.buffer_size = buffer_size
        self.lossfn = lossfn
        self.loss_grad = jax.value_and_grad(self.lossfn, 0)
        self.n_inner = n_inner
    

    def predict_obs(self, bel, X):
        yhat = bel.apply_fn(bel.params, X)
        return yhat

    def predict_state(self, bel, X):
        return bel
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_fifo_batches(self, ix):
        ix_lookback = (ix - self.buffer_size) + 1
        batches = jnp.arange(self.buffer_size) + ix_lookback
        return batches

    
    @partial(jax.jit, static_argnums=(0,))
    def _train_step(
        self,
        X: Float[Array, "num_obs dim_obs"],
        y: Float[Array, "num_obs dim_obs"],
        ixs: Int[Array, "batch_len"],
        state: TrainState,
    ) -> Tuple[float, TrainState]:
        """
        """
        loss, grads = self.loss_grad(state.params, X, y, ixs, state.apply_fn)
        state = state.apply_gradients(grads=grads)
        return loss, state

    @partial(jax.jit, static_argnums=(0,))
    def update_state(self, bel, X, y, ixs):
        def partial_step(_, bel):
            loss_train, bel = self._train_step(X, y, ixs, bel)
            return bel
        bel = jax.lax.fori_loop(0, self.n_inner - 1, partial_step, bel)
        loss, bel = self._train_step(X, y, ixs, bel)
        return loss, bel
    
    def scan(
        self,
        state: TrainState,
        X: Float[Array, "num_obs dim_obs"],
        y: Float[Array, "num_obs"],
        X_test: Union[None, Float[Array, "num_obs_test dim_obs"]] = None,
        y_test: Union[None, Float[Array, "num_obs_test"]] = None,
        loss_test_fn: Callable = None,
    ):
        num_obs = X.shape[0]

        def epoch_step(state, t):
            ixs = self._get_fifo_batches(t)
            loss_train, state = self.update_state(state, X, y, ixs)

            if (X_test is not None) and (y_test is not None):
                loss_test = loss_test_fn(state.params, X_test, y_test, state.apply_fn)
            else:
                loss_test = None

            losses = {
                "train": loss_train,
                "test": loss_test,
            } 
            return state, losses
        
        steps = jnp.arange(num_obs)
        state, losses = jax.lax.scan(epoch_step, state, steps)
        return state, losses

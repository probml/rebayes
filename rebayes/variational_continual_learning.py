"""
Coreset Variational Continual Learning
(Coreset VCL).

In this implementation, we consider a coreset (buffer)
of size `buffer_size`. At each step, we update the coreset
with probability `p_replace`. We consider two versions of the
buffer:
1. FIFO buffer: we replace the oldest datapoint in the buffer
    with the new datapoint.
2. Random buffer: we replace a random datapoint in the buffer
    with the new datapoint.


[2] Farquhar, S., Osborne, M., & Gal, Y. (2019).
    Radial Bayesian Neural Networks: Beyond Discrete Support
    In Large-Scale Bayesian Deep Learning. doi:10.48550/ARXIV.1907.00865
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
from functools import partial
from rebayes.base import Rebayes
from jax.flatten_util import ravel_pytree
from jaxtyping import Float, Array, Int
from flax.training.train_state import TrainState

BatchDense = nn.vmap(
    nn.Dense,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
)

def get_leaves(params):
    flat_params, _ = ravel_pytree(params)
    return flat_params

class VCLState(TrainState):
    """
    Coreset VCL state. In this implementation,
    we update the coreset (buffer) with probability
    and replace the oldest datapoint in the buffer
    if its full.

    We take the MFVI assumption over the parameters
    """
    buffer_size: int
    num_obs: int
    buffer_X: Float[Array, "buffer_size dim_features"]
    buffer_y: Float[Array, "buffer_size dim_output"]
    counter: Int[Array, "buffer_size"]
    prior_mean: Float[Array, "dim_params"]
    prior_std: Float[Array, "dim_params"]
    p_replace: float = 1.0

    def _update_buffer(self, step, buffer, item):
        ix_buffer = step % self.buffer_size
        buffer = buffer.at[ix_buffer].set(item)
        return buffer
    
    @staticmethod
    def _transform(eps, params):
        mean, rho = params
        std = jnp.log(1 + jnp.exp(rho))
        weight = mean + std * eps
        return weight
    
    def _sample_rbnn_params(self, scale=1.0):
        """
        Sample from a radial Bayesian neural network
        radial BNN of [2]. We modify the definition of the
        RBNN to include a scale parameter, which allows us
        to control the prior uncertainty over the posterior predictive.
        """
        key = jax.random.fold_in(self.key, self.step)
        key_eps, key_rho = jax.random.split(key)
        num_params = len(get_leaves(state.mean))

        # The radial dimension.
        r = jax.random.normal(key_rho) * scale

        eps = jax.random.normal(key_eps, (num_params,))
        eps = eps / jnp.linalg.norm(eps) * r
        eps = self.reconstruct_fn(eps)

        params = jax.tree_map(self._transform, eps, self.params)
        return params
    
    def apply_buffers(self, X, y):
        n_count = self.num_obs
        buffer_X = self._update_buffer(n_count, self.buffer_X, X)
        buffer_y = self._update_buffer(n_count, self.buffer_y, y)
        counter = self._update_buffer(n_count, self.counter, 1.0)
        
        return self.replace(
            num_obs=n_count + 1,
            buffer_X=buffer_X,
            buffer_y=buffer_y,
            counter=counter,
        )

        
    @classmethod
    def create(cls, *, apply_fn, params, tx,
               buffer_size, dim_features, dim_output,
               prior_mean, prior_std, **kwargs):
        """
        Here, params refers to the parameters of the
        variational distribution for non-coreset datapoint.
        """
        opt_state = tx.init(params)
        buffer_X = jnp.empty((buffer_size, dim_features))
        buffer_y = jnp.empty((buffer_size, dim_output))
        counter = jnp.zeros(buffer_size)
        
        return cls(
            step=0,
            num_obs=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            buffer_size=buffer_size,
            buffer_X=buffer_X,
            buffer_y=buffer_y,
            counter=counter,
            prior_mean=prior_mean,
            prior_std=prior_std,
            **kwargs
        )


class CoresetVCL(Rebayes):
    """
    FIFO Replay-buffer SGD training procedure
    """
    def __init__(self, lossfn, n_inner=1):
        self.lossfn = lossfn
        self.loss_grad = jax.value_and_grad(self.lossfn, 0)
        self.n_inner = n_inner

    def init_bel(self):
        raise NotImplementedError

    def predict_obs(self, bel, X):
        """
        Predict samples from the posterior predictive
        distribution
        """
        raise NotImplementedError

    def predict_state(self, bel):
        """
        The final variational distribution is given
        by (3) in Algorithm 1 of the paper.
        """
        raise NotImplementedError
        X, y = bel.buffer_X, bel.buffer_y
        bel = self._update_state_true(bel, X, y)
        return bel


    @partial(jax.jit, static_argnums=(0,))
    def _train_step(
        self,
        state: VCLState,
        X: Float[Array, "dim_features"],
        y: Float[Array, "dim_output"],
    ) -> Tuple[float, VCLState]:
        """
        """
        loss, grads = self.loss_grad(state.params, state.counter, X, y, state.apply_fn)
        state = state.apply_gradients(grads=grads)
        return loss, state

    def _update_state_true(self, bel, Xt, yt):
        """
        Parameter propagation step--we update the parameters
        only if Dt=(Xt, yt) is not included in the coreset.
        """
        def partial_step(_, bel):
            _, bel = self._train_step(bel, Xt, yt)
            return bel

        bel = jax.lax.fori_loop(0, self.n_inner, partial_step, bel)
        return bel

    def _update_state_false(self, bel, Xt, yt):
        """
        Coreset update step--we update the coreset and
        do not update the parameters.
        """
        bel = bel.apply_buffers(Xt, yt) 
        return bel

    @partial(jax.jit, static_argnums=(0,))
    def update_state(self, bel, Xt, yt):
        keyt = jax.random.fold_in(bel.key, bel.step)
        accept = jax.random.bernoulli(keyt, bel.p_replace)
        operands = (bel, Xt, yt)
        bel = jax.lax.cond(
            accept, self._update_state_true, self._update_state_false, operands
        )
        return bel

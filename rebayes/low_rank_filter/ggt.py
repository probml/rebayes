from functools import partial
from typing import Any, Callable, Union

import chex
from jax import jit, grad
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from rebayes.base import Rebayes, RebayesParams


@chex.dataclass
class GGTBel:
    mean: chex.Array
    gradients: chex.Array
    prev_grad = chex.Array
    num_obs: int = 0
    
    def update_buffer(self, g, beta):
        _, r = self.gradients.shape
        ix_buffer = self.num_obs % r
        gradients = beta * self.gradients
        gradients = gradients.at[:, ix_buffer].set(g)
        
        return self.replace(
            gradients = gradients,
            num_obs = self.num_obs + 1,
        )
    
    
@chex.dataclass
class GGTParams:
    initial_mean: Float[Array, "state_dim"]
    apply_fn: Callable
    loss_fn: Callable
    memory_size: int
    learning_rate: float
    beta1: float = 0.9 # momentum term
    beta2: float = 1.0 # forgetting term
    eps: float = 1e-4


class RebayesGGT(Rebayes):
    def __init__(
        self,
        params: GGTParams,
    ):
        super.__init__(params)
    
    def init_bel(self) -> GGTBel:
        m0 = self.params.initial_mean
        d, r = m0.shape[0], self.params.memory_size
        
        bel = GGTBel(
            mean = m0,
            gradients = jnp.zeros((d, r)),
            prev_grad = jnp.zeros(d),
        )
        
        return bel
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self,
        bel: GGTBel,
        X: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim"], Any]:
        y_pred = jnp.atleast_1d(self.params.apply_fn(bel.mean, X))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: GGTBel,
        X: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> GGTBel:
        g_prev = bel.prev_grad
        eta, eps = self.params.learning_rate, self.params.eps
        
        g = grad(self.params.loss_fn)(bel.mean, X, y) + self.params.beta1 * g_prev
        bel = bel.update_buffer(g, self.params.beta2)
        G = bel.gradients

        V, S, _ = jnp.linalg.svd(G.T @ G, full_matrices=False, hermition=True)
        Sig = jnp.sqrt(S)
        U = G @ (V * 1/Sig)
        update = g/eps + (U * (1/(Sig + eps) - 1/eps)) @ (U.T @ g)
        
        bel_post = bel.replace(
            mean = bel.mean - eta * update,
            prev_grad = g,
        )
        
        return bel_post
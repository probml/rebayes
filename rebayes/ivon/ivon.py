from typing import Callable, Union, Tuple, Any

import chex
from functools import partial
import jax
from jax import grad, jit, vmap
from jax.flatten_util import ravel_pytree
from jax.lax import scan
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import tensorflow_probability.substrates.jax as tfp

from rebayes.base import (
    CovMat,
    EmissionDistFn,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    FnStateToEmission,
    FnStateToEmission2,
    FnStateToState,
    Rebayes,
)

tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL
MVD = tfd.MultivariateNormalDiag


@chex.dataclass
class IVONBel:
    mean: chex.Array
    hessian: chex.Array
    grad: chex.Array
    count: int
    

class RebayesIVON(Rebayes):
    def __init__(
        self,
        apply_fn: Callable = None,
        loss_fn: Callable = None,
        n_sample: int = 1,
        beta_1: float = 0.9,
        beta_2: float = 1.0-1e-4,
        learning_rate: float = 2e-3,
        lamb: int = 1e3,
        weight_decay: float = 1e-3,
        rescale_lr: bool = True
    ):
        self.apply_fn = apply_fn
        self.loss_fn = lambda params, x, y: \
            jnp.mean(loss_fn(apply_fn(params, x), y))
        self.n_sample = n_sample
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lr = learning_rate
        self.lamb = lamb
        self.delta = weight_decay
        self.rescale_lr = rescale_lr

    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_hessian: float,
        Xinit: Float[Array, "input_dim"]=None,
        Yinit: Float[Array, "output_dim"]=None,
    ) -> IVONBel:
        if self.rescale_lr:
            self.alpha = self.lr * (initial_hessian + self.delta)
        bel = IVONBel(
            mean = initial_mean,
            hessian = initial_hessian * jnp.ones(initial_mean.shape),
            grad = jnp.zeros(initial_mean.shape),
            count = 0,
        )
        if Xinit is not None: # warmup sequentially
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
            
        return bel
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: IVONBel, 
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim"]:
        yhat = self.apply_fn(bel.mean, x)
        
        return yhat

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: IVONBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> IVONBel:
        m, H, g, count = bel.mean, bel.hessian, bel.grad, bel.count
        count += 1
        key = jax.random.PRNGKey(count)
        sigma = 1 / jnp.sqrt(self.lamb * (H + self.delta))
        
        def loss_gradient(theta):
            g = grad(self.loss_fn)(theta, x, y)
            return g
        
        noise = jax.random.normal(
            key, shape=(self.n_sample, m.shape[0])
        ) * sigma
        thetas = m + noise
        
        # Compute average gradient
        curr_grad = jnp.mean(vmap(loss_gradient)(thetas), axis=0)
        
        # Compute average Hessian
        curr_hessian = curr_grad * noise
        
        # Compute posterior
        g_cond = self.beta_1 * g + (1 - self.beta_1) * curr_grad
        g_cond = g_cond.reshape(g.shape)
        f = curr_hessian * (H + self.delta)
        H_cond = self.beta_2 * H + (1.0 - self.beta_2) * f + \
            (0.5 * (1. - self.beta_2)**2) * (H - f)**2 / (H + self.delta)
        H_cond = H_cond.reshape(H.shape)
        g_debiased = g_cond / (1 - self.beta_1**count)
        m_cond = m - self.alpha * (g_debiased + self.delta * m) / \
            (H_cond + self.delta)
        m_cond = m_cond.reshape(m.shape)
        
        
        bel_cond = IVONBel(
            mean = m_cond,
            hessian = H_cond,
            grad = g_cond,
            count = count,
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self,
        bel: IVONBel,
        key: Array,
        n_samples: int=100,
    ) -> Float[Array, "n_samples state_dim"]:
        sigma = 1 / jnp.sqrt(self.lamb * (bel.hessian + self.delta))
        noises = jax.random.normal(
            key, shape=(n_samples, bel.mean.shape[0])
        ) * sigma
        states = bel.mean + noises
        
        return states
    
    def scan(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_hessian: float,
        X: Float[Array, "batch_dim input_dim"],
        Y: Float[Array, "batch_dim output_dim"],
        callback=None,
        bel: IVONBel=None,
        **kwargs
    ) -> Tuple[IVONBel, Array]:
        num_timesteps = X.shape[0]
        self.lamb = num_timesteps
        def step(bel, t):
            x, y = X[t], Y[t]
            bel = self.update_state(bel, x, y)
            out = None
            if callback is not None:
                out = callback(bel, x, y, **kwargs)
                
            return bel, out
        carry = bel
        if carry is None:
            carry = self.init_bel(initial_mean, initial_hessian)
        bel, outputs = scan(step, carry, jnp.arange(num_timesteps))
        
        return bel, outputs
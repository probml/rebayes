import jax
import chex
import jax.numpy as jnp
import flax.linen as nn
from rebayes import base
from typing import Union
from jaxtyping import Array, Float
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree


class SubspaceModule(nn.Module):
    dim_subspace: int
    base_module: nn.Module
    init_normal: Callable = nn.initializers.normal()
    init_proj: Callable = nn.initializers.normal()


    def setup(self, X_buff, y_buff):
        # TODO:
        # Define a projection matrix, a bias vector and a reconstruction function
        ...

    def __call__(self, x):
        params = self.projection_matrix @ self.w + self.b
        params = self.reconstruct_params(params)
        yhat = self.base_module.apply(params, x)
        return yhat

class SubspaceBel(TrainState):
    projection_matrix: Float[Array, "dim_full dim_subspace"]

    @classmethod
    def create(cls, *, apply_fn, mean, covariance, projection, **kwargs):
        ...


class Subspace(base.Rebayes):
    def __init__(
        self,
        emission_mean_function
    ):
        self.emission_mean_function = emission_mean_function
        raise NotImplementedError

    def init_bel(self, initial_mean, initial_covariance, X, y):
        ...

    def update_state():
        ...
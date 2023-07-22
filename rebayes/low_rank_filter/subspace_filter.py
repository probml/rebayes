import jax
import chex
import jax.numpy as jnp
import flax.linen as nn
from rebayes import base
from typing import Union, Callable
from jaxtyping import Array, Float
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree


def subcify(cls):
    class SubspaceModule(nn.Module):
        dim_subspace: int
        init_normal: Callable = nn.initializers.normal()
        init_proj: Callable = nn.initializers.normal()


        # How to get the dim_subspace and dim_full?
        def setup(self):
            self.subspace = self.param(
                "subspace",
                self.init_proj,
                (dim_subspace,)
            )
            self.projection = self.param(
                "projection",
                self.init_proj,
                (dim_full, dim_subspace)
            )

        @nn.compact
        def __call__(self, x):
            params = self.projection @ self.subspace
            params = rfn(params)
            return NNet().apply(params, x)


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
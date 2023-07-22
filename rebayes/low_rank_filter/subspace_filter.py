import jax
import chex
import jax.numpy as jnp
import flax.linen as nn
from rebayes import base
from typing import Union, Callable
from jaxtyping import Array, Float
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree


def get_fparams(cls, X):
    dummy = cls(None, name="d")
    dummy_params = dummy.init(jax.random.PRNGKey(0), X)
    dummy_params, _ = ravel_pytree(dummy_params)
    return len(dummy_params)


def subcify(cls):
    class SubspaceModule(nn.Module):
        dim_subspace: int
        init_normal: Callable = nn.initializers.normal()
        init_proj: Callable = nn.initializers.normal()


        def init(self, rngs, X, *args, **kwargs):
            dim_full = get_fparams(cls, X)
            self.dim_full = dim_full
            return self.parent().init(self, rngs, *args, **kwargs)

        def setup(self):
            self.subspace = self.param(
                "subspace",
                self.init_proj,
                (self.dim_subspace,)
            )
            self.projection = self.param(
                "projection",
                self.init_proj,
                (self.dim_full, self.dim_subspace)
            )

        @nn.compact
        def __call__(self, x):
            params = self.projection @ self.subspace
            params = rfn(params)
            return cls().apply(params, x)
    
    return SubspaceModule


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
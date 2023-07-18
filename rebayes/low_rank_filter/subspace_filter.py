import jax
import chex
import jax.numpy as jnp
from rebayes import base
from typing import Union
from jaxtyping import Array, Float
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree


# TODO: Create flax classmodel that outputs parameters and a projection matrix

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
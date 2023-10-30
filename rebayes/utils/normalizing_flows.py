from functools import partial

from flax import linen as nn
import jax
from jax import jacrev, jacfwd, jit, lax, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
from matplotlib import animation
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf

tfd = tfp.distributions
tfb = tfp.bijectors


class NF_MLP(nn.Module):
    n_units: int=128
    n_layers: int=2
                        # create a Flax Module dataclass
    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        z = x
        for _ in range(self.n_layers):
            z = nn.Dense(self.n_units)(z)
            z = nn.relu(z)
        z = nn.Dense(x.shape[0]*2)(z)       # shape inference

        return z

def generate_shift_and_log_scale_fn(apply_fn, params):
    def shift_and_log_scale_fn(x, *args, **kwargs):
        result = apply_fn(params, x)
        shift, log_scale = jnp.split(result, 2, axis=-1)

        return shift, log_scale

    return shift_and_log_scale_fn


def construct_bijector(apply_fn, params, power):
    sl_fn = generate_shift_and_log_scale_fn(apply_fn, params)
    bijector = tfb.RealNVP(
        fraction_masked=0.5*(-1)**power,
        shift_and_log_scale_fn=sl_fn
    )

    return bijector


def construct_flow(apply_fn, params_stack):
    n, *_ = params_stack.shape
    bijector = []
    for i in range(n):
        bijector.append(construct_bijector(apply_fn, params_stack[i], i))
    bijector = tfb.Chain(bijector)

    return bijector


def init_normalizing_flow(model, input_dim, n_layers=4, key=0):
    input_dim = int(input_dim)
    assert input_dim % 2 == 0 # Even number of input dimensions
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, n_layers)

    params_stack, bijectors = [], []

    for i in range(n_layers):
        input = jnp.zeros(input_dim//2)
        params = model.init(keys[i], input)
        flat_params, unflatten_fn = ravel_pytree(params)
        params_stack.append(flat_params)
        apply_fn = lambda w, x: \
            model.apply(unflatten_fn(w), x)
        sl_fn = generate_shift_and_log_scale_fn(apply_fn, flat_params)
        bijector = tfb.RealNVP(
            fraction_masked=0.5*(-1)**i,
            shift_and_log_scale_fn=sl_fn
        )
        bijectors.append(bijector)

    params_stack = jnp.stack(params_stack)
    bijector = tfb.Chain(bijectors)

    apply_fn = lambda w, x: \
        model.apply(unflatten_fn(w), x)

    result = {
        "params": params_stack,
        "input_dim": input_dim,
        "n_layers": n_layers,
        "apply_fn": apply_fn,
        "bijector": bijector
    }

    return result
import jax
import optax
import distrax

import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

from typing import Callable
from functools import partial
from jax.flatten_util import ravel_pytree
from flax.training.train_state import TrainState

from rebayes.low_rank_filter import lofi
from rebayes.utils import callbacks
from rebayes.utils.utils import tree_to_cpu
from rebayes.sgd_filter import sgd
from rebayes.sgd_filter import replay_sgd as rsgd
from rebayes.datasets import rotating_mnist_data as rmnist

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


class MLP(nn.Module):
    n_out: int = 1
    n_hidden: int = 100
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_out, name="last-layer")(x)
        return x


def damp_angle(n_configs, minangle, maxangle):
    t = np.linspace(0, 1.5, n_configs)
    # angles = np.exp(t) * np.sin(35 * t)
    # angles = np.sin(35 * t) * (angles + 1) / 2 * (maxangle - minangle) + minangle + np.random.randn(n_configs) * 2
        
    # angles = np.random.randn(n_configs) * 50 + (maxangle + minangle) / 2
    angles = np.random.uniform(minangle, maxangle, size=n_configs)
    return angles


# @partial(jax.jit, static_argnames=("apply_fn",))
def log_likelihood(params, X, y, apply_fn, scale):
    y = y.ravel()
    mean = apply_fn(params, X).ravel()
    ll = distrax.Normal(mean, scale).log_prob(y)
    return ll.sum()
    

# @partial(jax.jit, static_argnames=("apply_fn",))
def lossfn(params, counter, X, y, apply_fn, scale):
    """
    Lossfunction for regression problems.
    """
    params_flat, _ = ravel_pytree(params)
    
    yhat = apply_fn(params, X).ravel()
    y = y.ravel()
    
    log_likelihood = distrax.Normal(yhat, scale).log_prob(y)
    log_likelihood = (log_likelihood * counter).sum()
    
    return -log_likelihood.sum()


def load_data():
    num_train = None
    frac_train = 1.0
    target_digit = 2

    np.random.seed(314)
    data = rmnist.load_and_transform(
        damp_angle, target_digit, num_train, frac_train, sort_by_angle=False
    )

    return data


if __name__ == "__main__":
    model = MLP()
    key = jax.random.PRNGKey(314)

    data = load_data()
    ymean, ystd = data["ymean"], data["ystd"]
    X_train, Y_train, labels_train = data["dataset"]["train"]
    X_test, Y_test, labels_test = data["dataset"]["test"]

    initial_covariance = 1 / 2000
    dynamics_weights = 1.0
    dynamics_covariance = 1e-7
    emission_cov = 0.01
    memory_size = 10
    scale = np.sqrt(emission_cov)
    tx = optax.adam(1e-4)

    part_lossfn = partial(lossfn, scale=scale)
    part_lossfn = jax.jit(part_lossfn, static_argnames=("apply_fn",))
    part_log_likelihood = partial(log_likelihood, scale=scale)
    part_log_likelihood = jax.jit(part_log_likelihood, static_argnames=("apply_fn",))

    agent_lofi, rfn = lofi.init_regression_agent(
        key, model, X_train,
        initial_covariance, dynamics_weights, dynamics_covariance,
        emission_cov, memory_size
    )

    agent_rsgd = rsgd.init_regression_agent(
        key, part_log_likelihood, model, X_train, tx, memory_size,
        lossfn=part_lossfn,
        prior_precision=1 / initial_covariance,
    )

    callback = partial(callbacks.cb_reg_mc,
                    ymean=ymean, ystd=ystd,
                    X_test=X_test, y_test=Y_test,
                    key=key,
    )

    callback_lofi = partial(callback, apply_fn=agent_lofi.params.emission_mean_function, agent=agent_lofi)
    callback_rsgd = partial(callback, apply_fn=agent_rsgd.apply_fn, agent=agent_rsgd)

    bel_rsgd, output_rsgd = agent_rsgd.scan(X_train, Y_train, progress_bar=True, callback=callback_rsgd)
    output_rsgd = tree_to_cpu(output_rsgd)

    bel_lofi, output_lofi = agent_lofi.scan(X_train, Y_train, progress_bar=True, callback=callback_lofi)
    output_lofi = tree_to_cpu(output_lofi)


    print("Done!")

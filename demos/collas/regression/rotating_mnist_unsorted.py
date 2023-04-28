import jax
import optax
import distrax
import numpy as np
import flax.linen as nn
import tensorflow_probability.substrates.jax as tfp

from typing import Callable
from functools import partial
from rebayes.utils import callbacks
from rebayes.low_rank_filter import lofi
from bayes_opt import BayesianOptimization
from rebayes.utils.utils import tree_to_cpu
from rebayes.sgd_filter import replay_sgd as rsgd
from rebayes.datasets import rotating_mnist_data as rmnist
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


def uniform_angles(n_configs, minangle, maxangle):
    angles = np.random.uniform(minangle, maxangle, size=n_configs)
    return angles


def increase_damp_angle(n_configs, minangle, maxangle):
    t = np.linspace(0, 1.5, n_configs)
    angles = np.exp(t) * np.sin(35 * t)
    angles = (angles + 1) / 2 * (maxangle - minangle) + minangle + np.random.randn(n_configs) * 2
    return angles


def log_likelihood(params, X, y, apply_fn, scale):
    y = y.ravel()
    mean = apply_fn(params, X).ravel()
    ll = distrax.Normal(mean, scale).log_prob(y)
    return ll.sum()


def lossfn(params, counter, X, y, apply_fn, scale):
    """
    Lossfunction for regression problems.
    """
    yhat = apply_fn(params, X).ravel()
    y = y.ravel()

    log_likelihood = distrax.Normal(yhat, scale).log_prob(y)
    log_likelihood = (log_likelihood * counter).sum()

    return -log_likelihood.sum()


def load_data(data_transform):
    num_train = None
    frac_train = 1.0
    target_digit = 2

    np.random.seed(314)
    data = rmnist.load_and_transform(
        data_transform, target_digit, num_train, frac_train, sort_by_angle=False
    )

    return data


if __name__ == "__main__":
    import sys
    model = MLP()
    key = jax.random.PRNGKey(314)

    _, problem_type = sys.argv
    match problem_type:
        case "uniform":
            loadfn = uniform_angles
        case "increase":
            loadfn = increase_damp_angle
        case _:
            raise ValueError(f"Unknown problem type {problem_type}")

    data = load_data(loadfn)
    ymean, ystd = data["ymean"], data["ystd"]
    X_train, Y_train, labels_train = data["dataset"]["train"]
    X_test, Y_test, labels_test = data["dataset"]["test"]

    initial_covariance = 1 / 2000
    emission_cov = 0.01
    # dynamics_weights = 1.0
    # dynamics_covariance = 1e-7
    memory_size = 10
    scale = np.sqrt(emission_cov)

    part_lossfn = partial(lossfn, scale=scale)
    part_lossfn = jax.jit(part_lossfn, static_argnames=("apply_fn",))
    part_log_likelihood = partial(log_likelihood, scale=scale)
    part_log_likelihood = jax.jit(part_log_likelihood, static_argnames=("apply_fn",))

    n_callback = 500
    X_callback = X_train[:n_callback]
    Y_callback = Y_train[:n_callback]
    X_train = X_train[n_callback:]
    Y_train = Y_train[n_callback:]

    callback = partial(callbacks.cb_reg_mc,
                    ymean=ymean, ystd=ystd,
                    X_test=X_callback, y_test=Y_callback,
                    key=key,
    )

    def bbf_lofi(log_1m_dynamics_weights, log_dynamics_covariance, memory_size):
        dynamics_weights = 1 - np.exp(log_1m_dynamics_weights)
        dynamics_covariance = np.exp(log_dynamics_covariance)
        agent_lofi, rfn = lofi.init_regression_agent(
            key, model, X_train,
            initial_covariance, dynamics_weights, dynamics_covariance,
            emission_cov, memory_size
        )

        callback_lofi = partial(callback, apply_fn=agent_lofi.params.emission_mean_function, agent=agent_lofi)

        _, output_lofi = agent_lofi.scan(X_train, Y_train, progress_bar=False, callback=callback_lofi)
        output_lofi = tree_to_cpu(output_lofi)

        metric = output_lofi["nlpd_test"][-1]
        isna = np.isnan(metric)
        metric = 1e10 if isna else metric
        return -metric

    def bbf_rsgd(log_lr, tx_fn, memory_size):
        lr = np.exp(log_lr)
        tx = tx_fn(lr)

        agent_rsgd = rsgd.init_regression_agent(
            key, part_log_likelihood, model, X_train, tx, memory_size,
            lossfn=part_lossfn,
            prior_precision=1 / initial_covariance,
        )

        callback_rsgd = partial(callback, apply_fn=agent_rsgd.apply_fn, agent=agent_rsgd)
        _, output_rsgd = agent_rsgd.scan(X_train, Y_train, progress_bar=False, callback=callback_rsgd)
        output_rsgd = tree_to_cpu(output_rsgd)
        metric = output_rsgd["nlpd_test"][-1]
        isna = np.isnan(metric)
        metric = 1e10 if isna else metric
        return -metric

    random_state = 2718

    bounds_rsgd = {
        "log_lr": (-15, -6.5)
    }

    bounds_lofi = {
        "log_1m_dynamics_weights": (-20, -6.5),
        "log_dynamics_covariance": (-15, -6.5),
    }

    memory_rsgd_list = [1, 5, 10]
    adam_optimisers = {}
    for memory_size in memory_rsgd_list:
        optimiser_rsgd_adam = BayesianOptimization(
            f=partial(bbf_rsgd, tx_fn=optax.adam, memory_size=memory_size),
            pbounds=bounds_rsgd,
            allow_duplicate_points=True,
            random_state=random_state,
        )
        adam_optimisers[memory_size] = optimiser_rsgd_adam

    rsgd_optimisers = {}
    for memory_size in memory_rsgd_list:
        optimiser_rsgd = BayesianOptimization(
            f=partial(bbf_rsgd, tx_fn=optax.sgd, memory_size=memory_size),
            pbounds=bounds_rsgd,
            allow_duplicate_points=True,
            random_state=random_state,
        )
        rsgd_optimisers[memory_size] = optimiser_rsgd

    memory_lofi = [5, 10]
    lofi_optimisers = {}
    for memory_size in memory_lofi:
        optimiser_lofi = BayesianOptimization(
            f=partial(bbf_lofi, memory_size=memory_size),
            pbounds=bounds_lofi,
            allow_duplicate_points=True,
            random_state=random_state,
        )
        lofi_optimisers[memory_size] = optimiser_lofi

    optimizer_eval_kwargs = {
        "init_points": 10,
        "n_iter": 15,
    }

    print("Training RSGD")
    for memory_size, optimiser_rsgd in rsgd_optimisers.items():
        print(f"Memory size: {memory_size}")
        optimiser_rsgd.maximize(**optimizer_eval_kwargs)
    print("Training RSGD adam")
    for memory_size, optimiser_rsgd_adam in adam_optimisers.items():
        print(f"Memory size: {memory_size}")
        optimiser_rsgd_adam.maximize(**optimizer_eval_kwargs)
    print("Training LoFi")
    for memory_size, optimiser_lofi in lofi_optimisers.items():
        print(f"Memory size: {memory_size}")
        optimiser_lofi.maximize(**optimizer_eval_kwargs)

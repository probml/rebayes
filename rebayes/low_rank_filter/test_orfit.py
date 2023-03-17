import pytest
import numpy as np
import jax.numpy as jnp
import time

from rebayes.utils import rotating_mnist_data
from rebayes.utils.utils import get_mlp_flattened_params
from rebayes.low_rank_filter.orfit import RebayesORFit
from rebayes.extended_kalman_filter.ekf import RebayesEKF
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.generalized_gaussian_ssm.inference import EKFIntegrals, conditional_moments_gaussian_filter
from rebayes.base import RebayesParams


def allclose(u, v):
    return jnp.allclose(u, v, atol=1e-3)


def load_rmnist_data(num_train=100):
    # Load rotated MNIST dataset
    np.random.seed(314)

    train, _ = rotating_mnist_data.load_rotated_mnist(target_digit=2)
    X_train, y_train = train

    X_train = jnp.array(X_train)[:num_train]
    y_train = jnp.array(y_train)[:num_train]

    X_train = (X_train - X_train.mean()) / X_train.std()

    return X_train, y_train


def setup_orfit(memory_size):
    # Define Linear Regression as single layer perceptron
    input_dim, hidden_dims, output_dim = 784, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

    # Define ORFit parameters
    model_params = RebayesParams(
        initial_mean = flat_params,
        initial_covariance = None,
        dynamics_weights = None,
        dynamics_covariance = None,
        emission_mean_function = apply_fn,
        emission_cov_function = None,
    )
    orfit_estimator = RebayesORFit(
        params = model_params,
        memory_size = memory_size,
    ) 

    return orfit_estimator


def setup_ekf():
    # Define Linear Regression as single layer perceptron
    input_dim, hidden_dims, output_dim = 784, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)

    # Define Kalman Filter parameters
    model_params = RebayesParams(
        initial_mean = flat_params,
        initial_covariance = jnp.eye(len(flat_params)),
        dynamics_weights = 1.0,
        dynamics_covariance = 0.0,
        emission_mean_function = apply_fn,
        emission_cov_function = lambda w, x: jnp.array([0.]),
    )
    ekf_estimator = RebayesEKF(
        params = model_params,
        method = "fcekf"
    )
    
    return ekf_estimator


def test_rebayes_orfit_loop():
    # Load rotated MNIST dataset
    n_train = 200
    X_train, y_train = load_rmnist_data(n_train)

    # Run Infinite-memory ORFit
    orfit_estimator = setup_orfit(n_train)
    orfit_before_time = time.time()
    orfit_bel = orfit_estimator.init_bel()
    for i in range(n_train):
        orfit_bel = orfit_estimator.update_state(orfit_bel, X_train[i], y_train[i])
    orfit_after_time = time.time()
    print(f"Looped ORFit took {orfit_after_time - orfit_before_time} seconds.")

    # Run Kalman Filter
    ekf_estimator = setup_ekf()
    ekf_before_time = time.time()
    ekf_bel = ekf_estimator.init_bel()
    for i in range(n_train):
        ekf_bel = ekf_estimator.update_state(ekf_bel, X_train[i], y_train[i])
    ekf_after_time = time.time()
    print(f"Kalman Filter took {ekf_after_time - ekf_before_time} seconds.")

    assert allclose(orfit_bel.mean, ekf_bel.mean)


def test_rebayes_orfit_scan():
    # Load rotated MNIST dataset
    n_train = 200
    X_train, y_train = load_rmnist_data(n_train)

    # Define mean callback function
    def callback(bel, *args, **kwargs):
        return bel.mean

    # Run Infinite-memory ORFit
    orfit_estimator = setup_orfit(n_train)
    orfit_before_time = time.time()
    _, orfit_outputs = orfit_estimator.scan(X_train, y_train, callback)
    orfit_after_time = time.time()
    print(f"Scanned ORFit took {orfit_after_time - orfit_before_time} seconds.")

    # Run Kalman Filter
    ekf_estimator = setup_ekf()
    ekf_before_time = time.time()
    _, ekf_outputs = ekf_estimator.scan(X_train, y_train, callback)
    ekf_after_time = time.time()
    print(f"Kalman Filter took {ekf_after_time - ekf_before_time} seconds.")

    assert allclose(orfit_outputs, ekf_outputs)


from jaxtyping import Float, Array
from typing import Callable, NamedTuple, Union, Tuple, Any
from functools import partial
import chex
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jacfwd, vmap, grad, jit
from jax.tree_util import tree_map, tree_reduce
from jax.flatten_util import ravel_pytree

import flax
import flax.linen as nn
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from collections import namedtuple
from itertools import cycle

from dynamax.linear_gaussian_ssm import LinearGaussianSSM

from rebayes.utils.utils import get_mlp_flattened_params
from rebayes.extended_kalman_filter.ekf import RebayesEKF
from rebayes.dual_filter.dual_estimator import rebayes_scan, DualBayesParams, ObsModel
from rebayes.dual_filter.dual_ekf import make_dual_ekf_estimator, EKFParams

def allclose(u, v):
    return jnp.allclose(u, v, atol=1e-3)


def make_linreg_data():
    n_obs = 21
    x = jnp.linspace(0, 20, n_obs)
    X = x[:, None] # reshape to (T,1)
    y = jnp.array(
        [2.486, -0.303, -4.053, -4.336, -6.174, -5.604, -3.507, -2.326, -4.638, -0.233, -1.986, 1.028, -2.264,
        -0.451, 1.167, 6.652, 4.145, 5.268, 6.34, 9.626, 14.784])
    Y = y[:, None] # reshape to (T,1)
    return X, Y

def make_linreg_prior():
    obs_var = 0.1
    mu0 = jnp.zeros(2)
    Sigma0 = jnp.eye(2) * 1
    return (obs_var, mu0, Sigma0)

def make_linreg_dual_bayes(nfeatures):
    (obs_var, mu0, Sigma0) = make_linreg_prior()
    
    # Define Linear Regression as MLP with no hidden layers
    input_dim, hidden_dims, output_dim = nfeatures, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)
    nparams = len(flat_params)
    
    params = DualBayesParams(
        mu0=mu0,
        eta0=1/Sigma0[0,0],
        gamma = 1.0,
        q = 0.0,
        obs_noise_var = obs_var,
        alpha = 0,
    )
    obs_model = ObsModel(
        emission_mean_function = lambda w, x: apply_fn(w, x),
        emission_cov_function = lambda w, x: obs_var
    )

    return params, obs_model

    
def run_kalman(X, Y):
    N = X.shape[0]
    X1 = jnp.column_stack((jnp.ones(N), X))  # Include column of 1s
    (obs_var, mu0, Sigma0) = make_linreg_prior()
    nfeatures = X1.shape[1]
    # we use H=X1 since z=(b, w), so z'u = (b w)' (1 x)
    lgssm = LinearGaussianSSM(state_dim = nfeatures, emission_dim = 1, input_dim = 0)
    F = jnp.eye(nfeatures) # dynamics = I
    Q = jnp.zeros((nfeatures, nfeatures))  # No parameter drift.
    R = jnp.ones((1, 1)) * obs_var

    params, _ = lgssm.initialize(
        initial_mean=mu0,
        initial_covariance=Sigma0,
        dynamics_weights=F,
        dynamics_covariance=Q,
        emission_weights=X1[:, None, :], # (t, 1, D) where D = num input features
        emission_covariance=R,
        )
    lgssm_posterior = lgssm.filter(params, Y) 
    return lgssm_posterior



def test_linreg():
    (X, Y) = make_linreg_data()
    lgssm_posterior = run_kalman(X, Y)
    mu_kf = lgssm_posterior.filtered_means
    cov_kf = lgssm_posterior.filtered_covariances
    ll_kf = lgssm_posterior.marginal_loglik

    N,D = X.shape
    params, obs_model = make_linreg_dual_bayes(D)
    ekf_params = EKFParams(method="fcekf", obs_noise_var_lr=0)
    estimator = make_dual_ekf_estimator(params, obs_model, ekf_params)

    def callback(params, bel, pred_obs, t, u, y, bel_pred):
        m = pred_obs # estimator.predict_obs(params, bel_pred, u)
        P = estimator.predict_obs_cov(params, bel_pred, u)
        ll = MVN(m, P).log_prob(jnp.atleast_1d(y))
        return ll

    carry, lls = rebayes_scan(estimator,  X, Y, callback)
    params, final_bel = carry
    print(carry)
    T = mu_kf.shape[0]
    assert allclose(final_bel.mean, mu_kf[T-1])
    assert allclose(final_bel.cov, cov_kf[T-1])
    ll = jnp.sum(lls)
    assert jnp.allclose(ll, ll_kf, atol=1e-1)


if __name__ == "__main__":
    test_linreg()
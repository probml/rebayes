from functools import partial
from jax import jit
from jax import numpy as jnp
from jaxtyping import Float, Array

from rebayes.base import _jacrev_2d, Rebayes, RebayesParams, Gaussian
from rebayes.extended_kalman_filter.ekf_inference import (
    _full_covariance_condition_on,
    _fully_decoupled_ekf_condition_on,
    _variational_diagonal_ekf_condition_on,
)


class RebayesEKF(Rebayes):
    def __init__(
        self,
        params: RebayesParams,
        method: str
    ):
        self.params = params
        self.method = method
        if method not in ['fcekf', 'vdekf', 'fdekf']:
            raise ValueError('unknown method ', method)

    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        if self.method == 'fcekf':
            return super().predict_state(bel)

        # Diagonal EKF: assume that dynamics weights and covariance are given by 1d vector
        m, P = bel.mean, bel.cov 
        F = self.params.dynamics_weights
        Q = self.params.dynamics_covariance
        pred_mean = F * m
        pred_cov = F**2 * P + Q
        return Gaussian(mean=pred_mean, cov=pred_cov)

    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        if self.method == 'fcekf':
            return super().predict_obs(bel, u)
        
        # Diagonal EKF: assume that dynamics weights and covariance are given by 1d vector
        prior_mean, prior_cov = bel.mean, bel.cov 
        m_Y = lambda z: self.params.emission_mean_function(z, u)
        Cov_Y = lambda z: self.params.emission_cov_function(z, u)

        yhat = jnp.atleast_1d(m_Y(prior_mean))
        R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)

        Sigma_obs = (prior_cov * H) @ H.T + R
        return Gaussian(mean=yhat, cov=Sigma_obs)

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        if self.method == 'fcekf':
            self.update_fn = _full_covariance_condition_on
        elif self.method == 'vdekf':
            self.update_fn = _variational_diagonal_ekf_condition_on
        elif self.method == 'fdekf':
            self.update_fn = _fully_decoupled_ekf_condition_on
        m, P = bel.mean, bel.cov # p(z(t) | y(1:t-1))
        mu, Sigma = self.update_fn(m, P, self.params.emission_mean_function, self.params.emission_cov_function, u, y, num_iter=1)
        return Gaussian(mean=mu, cov=Sigma)
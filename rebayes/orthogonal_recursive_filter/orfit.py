"""
Implementation of variants of Orthogonal Recursive Fitting (ORFit) [1] algorithm for online learning.

[1] Min, Y., Ahn, K, & Azizan, N. (2022, July).
One-Pass Learning via Bridging Orthogonal Gradient Descent and Recursive Least-Squares.
Retrieved from https://arxiv.org/abs/2207.13853
"""

from functools import partial

import jax.numpy as jnp
from jax import jit
import chex

from rebayes.base import RebayesParams, Rebayes, Gaussian
from rebayes.orthogonal_recursive_filter.orfit_inference import (
    _jacrev_2d,
    ORFitParams,
    _orfit_condition_on,
    _generalized_orfit_condition_on,
    _generalized_orfit_predict,
    _generalized_orfit_condition_on_with_adaptive_observation_variance,
    _generalized_orfit_marginalize,
)


@chex.dataclass
class ORFitBel:
    mean: chex.Array
    basis: chex.Array
    sigma: chex.Array
    nu: float = None
    rho: float = None
    tau: float = None


class RebayesORFit(Rebayes):
    def __init__(
        self,
        model_params: RebayesParams,
        orfit_params: ORFitParams,
        method: str,
    ):
        self.method = method
        self.nu, self.rho, self.tau = None, None, None
        if method == 'orfit':
            pass
        elif method == 'generalized_orfit' or method == 'generalized_orfit_adaptive_obs_var':
            initial_cov = model_params.initial_covariance
            assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
            self.eta = 1/initial_cov
            self.gamma = model_params.dynamics_weights
            assert isinstance(self.gamma, float), "Dynamics decay term must be a scalar."
            self.q = (1 - self.gamma**2) / self.eta
            if method == 'generalized_orfit_adaptive_obs_var':
                self.nu, self.rho, self.tau = 0.0, 0.0, 0.0
        else:
            raise ValueError(f"Unknown method {method}.")
        self.model_params = model_params
        self.m, self.sv_threshold = orfit_params
        self.U0 = jnp.zeros((len(model_params.initial_mean), self.m))
        self.Sigma0 = jnp.zeros((self.m,))

    def init_bel(self):
        return ORFitBel(
            mean=self.model_params.initial_mean, basis=self.U0, sigma=self.Sigma0, 
            nu=self.nu, rho=self.rho, tau=self.tau
        )
    
    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, U, Sigma, nu, rho, tau = bel.mean, bel.basis, bel.sigma, bel.nu, bel.rho, bel.tau
        if self.method == 'orfit':
            return bel
        else:
            m_pred, Sigma_pred = _generalized_orfit_predict(m, Sigma, self.gamma, self.q)
            U_pred = U

        return ORFitBel(mean=m_pred, basis=U_pred, sigma=Sigma_pred, nu=nu, rho=rho, tau=tau)

    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        m, U = bel.mean, bel.basis
        m_Y = lambda z: self.model_params.emission_mean_function(z, u)
        Cov_Y = lambda z: self.model_params.emission_cov_function(z, u)
        
        y_pred = jnp.atleast_1d(m_Y(m))
        H =  _jacrev_2d(m_Y, m)
        Sigma_obs = H @ H.T - (H @ U) @ (H @ U).T

        if self.method == 'generalized_orfit':
            R = jnp.atleast_2d(Cov_Y(m))
            Sigma_obs += R
        
        return Gaussian(mean=y_pred, cov=Sigma_obs) # TODO: regression/classification separation (reg:mean / var, class: dist)

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        m, U, Sigma, nu, rho, tau = bel.mean, bel.basis, bel.sigma, bel.nu, bel.rho, bel.tau
        if self.method == 'orfit':
            m_cond, U_cond, Sigma_cond = _orfit_condition_on(
                m, U, Sigma, self.model_params.emission_mean_function, u, y, self.sv_threshold
            )
        elif self.method == 'generalized_orfit':
            m_cond, U_cond, Sigma_cond = _generalized_orfit_condition_on(
                m, U, Sigma, self.eta, self.model_params.emission_mean_function, 
                self.model_params.emission_cov_function, u, y, self.sv_threshold
            )
        elif self.method == 'generalized_orfit_adaptive_obs_var':
            m_cond, U_cond, Sigma_cond = _generalized_orfit_condition_on_with_adaptive_observation_variance(
                m, U, Sigma, self.eta, self.model_params.emission_mean_function, u, y, self.sv_threshold
            )
            nu, rho, tau = _generalized_orfit_marginalize(
                m_cond, U_cond, Sigma_cond, self.eta, self.model_params.emission_mean_function, u, y, nu, rho
            )    
        return ORFitBel(mean=m_cond, basis=U_cond, sigma=Sigma_cond, nu=nu, rho=rho, tau=tau)

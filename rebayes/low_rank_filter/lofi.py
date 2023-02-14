from functools import partial

import jax.numpy as jnp
from jax import jit
import chex

from rebayes.base import RebayesParams, Rebayes, Gaussian
from rebayes.low_rank_filter.lofi_inference import (
    _jacrev_2d,
    LoFiParams,
    _orfit_condition_on,
    _lofi_condition_on,
    _lofi_predict,
    _aov_lofi_condition_on,
    _lofi_marginalize,
    _full_svd_lofi_condition_on,
)


@chex.dataclass
class LoFiBel:
    mean: chex.Array
    basis: chex.Array
    sigma: chex.Array
    nu: float = None
    rho: float = None
    tau: float = None


class RebayesLoFi(Rebayes):
    def __init__(
        self,
        model_params: RebayesParams,
        orfit_params: LoFiParams,
        method: str,
    ):
        self.method = method
        self.nu, self.rho, self.tau = None, None, None
        if method == 'orfit':
            pass
        elif method == 'lofi' or method == 'aov_lofi' or method == 'full_svd_lofi':
            initial_cov = model_params.initial_covariance
            assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
            self.eta = 1/initial_cov
            self.gamma = model_params.dynamics_weights
            assert isinstance(self.gamma, float), "Dynamics decay term must be a scalar."
            self.q = (1 - self.gamma**2) / self.eta
            if method == 'aov_lofi' or method == 'full_svd_lofi':
                self.nu, self.rho, self.tau = 0.0, 0.0, 0.0
        else:
            raise ValueError(f"Unknown method {method}.")
        self.model_params = model_params
        self.m, self.sv_threshold = orfit_params
        self.U0 = jnp.zeros((len(model_params.initial_mean), self.m))
        self.Sigma0 = jnp.zeros((self.m,))

    def init_bel(self):
        return LoFiBel(
            mean=self.model_params.initial_mean, basis=self.U0, sigma=self.Sigma0, 
            nu=self.nu, rho=self.rho, tau=self.tau
        )
    
    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, U, Sigma, nu, rho, tau = bel.mean, bel.basis, bel.sigma, bel.nu, bel.rho, bel.tau
        if self.method == 'orfit':
            return bel
        else:
            m_pred, Sigma_pred = _lofi_predict(m, Sigma, self.gamma, self.q)
            U_pred = U

        return LoFiBel(mean=m_pred, basis=U_pred, sigma=Sigma_pred, nu=nu, rho=rho, tau=tau)

    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        m, U, sigma, tau = bel.mean, bel.basis, bel.sigma, bel.tau
        m_Y = lambda z: self.model_params.emission_mean_function(z, u)
        Cov_Y = lambda z: self.model_params.emission_cov_function(z, u)
        
        # Predicted mean
        y_pred = jnp.atleast_1d(m_Y(m))

        # Predicted covariance
        H =  _jacrev_2d(m_Y, m)
        if self.method == 'orfit':
            Sigma_obs = H @ H.T - (H @ U) @ (H @ U).T
        else:
            lamb = (sigma**2)/(self.eta**2 * jnp.ones(sigma.shape) + self.eta * sigma**2)
            HU = H @ U
            HSHT = H @ H.T/self.eta - (lamb * HU) @ (HU).T
            if self.method == 'lofi':
                R = jnp.atleast_2d(Cov_Y(m))
                Sigma_obs = HSHT + R
            else:
                Sigma_obs = tau * HSHT
        
        return Gaussian(mean=y_pred, cov=Sigma_obs) # TODO: regression/classification separation (reg:mean / var, class: dist)

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        m, U, Sigma, nu, rho, tau = bel.mean, bel.basis, bel.sigma, bel.nu, bel.rho, bel.tau
        if self.method == 'orfit':
            m_cond, U_cond, Sigma_cond = _orfit_condition_on(
                m, U, Sigma, self.model_params.emission_mean_function, u, y, self.sv_threshold
            )
        elif self.method == 'lofi':
            m_cond, U_cond, Sigma_cond = _lofi_condition_on(
                m, U, Sigma, self.eta, self.model_params.emission_mean_function, 
                self.model_params.emission_cov_function, u, y, self.sv_threshold
            )
        elif self.method == 'aov_lofi':
            m_cond, U_cond, Sigma_cond = _aov_lofi_condition_on(
                m, U, Sigma, self.eta, self.model_params.emission_mean_function, u, y, self.sv_threshold
            )
            nu, rho, tau = _lofi_marginalize(
                m_cond, U_cond, Sigma_cond, self.eta, self.model_params.emission_mean_function, u, y, nu, rho
            )    
        elif self.method == 'full_svd_lofi':
            m_cond, U_cond, Sigma_cond = _full_svd_lofi_condition_on(
                m, U, Sigma, self.eta, self.model_params.emission_mean_function, u, y, self.sv_threshold
            )
            nu, rho, tau = _lofi_marginalize(
                m_cond, U_cond, Sigma_cond, self.eta, self.model_params.emission_mean_function, u, y, nu, rho
            )
        return LoFiBel(mean=m_cond, basis=U_cond, sigma=Sigma_cond, nu=nu, rho=rho, tau=tau)

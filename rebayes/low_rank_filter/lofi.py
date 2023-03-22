from functools import partial
from typing import Union, Any

import chex
from jax import jit
import jax.numpy as jnp
from jaxtyping import Float, Array

from rebayes.base import RebayesParams, Rebayes, CovMat
from rebayes.low_rank_filter.lofi_core import (
    _jacrev_2d,
    _lofi_spherical_cov_inflate,
    _lofi_spherical_cov_predict,
    _lofi_estimate_noise,
    _lofi_spherical_cov_condition_on,
    _lofi_orth_condition_on,
    _lofi_diagonal_cov_inflate,
    _lofi_diagonal_cov_predict,
    _lofi_diagonal_cov_condition_on,
)


# Common classes ---------------------------------------------------------------

INFLATION_METHODS = [
    'bayesian', 
    'simple', 
    'hybrid',
]


@chex.dataclass
class LoFiBel:
    pp_mean: chex.Array
    mean: chex.Array
    basis: chex.Array
    svs: chex.Array
    eta: float
    gamma: float
    q: float
    
    Ups: CovMat = None
    nobs: int = 0
    obs_noise_var: float = 0.0

    # @property
    # def cov(self):
    #     """
    #     For large-dimensional systems,
    #     use at your own risk.
    #     """
    #     Lambda = self.singular_values
    #     num_features = len(self.mean)
    #     D = Lambda ** 2 / (self.eta * (self.eta + Lambda ** 2))
    #     D = jnp.diag(D)

    #     I = jnp.eye(num_features)
    #     cov = I / self.eta - self.basis @ D @ self.basis.T
    #     return cov


@chex.dataclass
class LoFiParams:
    """Lightweight container for LOFI parameters.
    """
    memory_size: int
    steady_state: bool = False
    inflation: str = 'bayesian'


@chex.dataclass
class PosteriorLoFiFiltered:
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]
    filtered_svs: Float[Array, "ntime memory_size"] = None


class RebayesLoFi(Rebayes):
    def __init__(
        self,
        model_params: RebayesParams,
        lofi_params: LoFiParams,
    ):
        # Check inflation type
        if lofi_params.inflation not in INFLATION_METHODS:
            raise ValueError(f"Unknown inflation method: {lofi_params.inflation}.")
        
        self.model_params = model_params
        self.lofi_params = lofi_params
        
    def init_bel(self):
        pp_mean = self.model_params.initial_mean # Predictive prior mean
        init_mean = self.model_params.initial_mean # Initial mean
        memory_size = self.lofi_params.memory_size
        init_basis = jnp.zeros((len(init_mean), memory_size)) # Initial basis
        init_svs = jnp.zeros(memory_size) # Initial singular values
        init_eta = 1 / self.model_params.initial_covariance # Initial precision
        gamma = self.model_params.dynamics_weights # Dynamics weights
        q = self.model_params.dynamics_covariance # Dynamics covariance
        if self.lofi_params.steady_state: # Steady-state constraint
            q = self.steady_state_constraint(init_eta, gamma)
        init_Ups = jnp.ones((len(init_mean), 1)) * init_eta
        
        return LoFiBel(
            pp_mean = pp_mean,
            mean = init_mean,
            basis = init_basis,
            svs = init_svs,
            eta = init_eta,
            gamma = gamma,
            q = q,
            Ups = init_Ups
        )

    @staticmethod
    def steady_state_constraint(
        eta: float, 
        gamma: float,
    ) -> float:
        """Return dynamics covariance according to the steady-state constraint."""
        q = (1 - gamma**2) / eta
        
        return q
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: LoFiBel, 
        x: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim"], Any]: 
        m = bel.mean
        m_Y = lambda z: self.model_params.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred


# Spherical LOFI ---------------------------------------------------------------

class RebayesLoFiSpherical(RebayesLoFi):
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: LoFiBel,
    ) -> LoFiBel:
        m0, m, U, Lambda, eta, gamma, q = \
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.gamma, bel.q
        alpha = self.model_params.dynamics_covariance_inflation_factor
        inflation = self.lofi_params.inflation
        
        # Inflate posterior covariance.
        m_infl, U_infl, Lambda_infl, eta_infl = \
            _lofi_spherical_cov_inflate(m0, m, U, Lambda, eta, alpha, inflation)
        
        # Predict dynamics.
        pp_mean_pred, m_pred, U_pred, Lambda_pred, eta_pred = \
            _lofi_spherical_cov_predict(m0, m_infl, U_infl, Lambda_infl, gamma, 
                                        q, eta_infl, self.lofi_params.steady_state)
        
        bel_pred = bel.replace(
            pp_mean = pp_mean_pred,
            mean = m_pred,
            basis = U_pred,
            svs = Lambda_pred,
            eta = eta_pred,
        )
        
        return bel_pred
    
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, eta, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.obs_noise_var
        m_Y = lambda z: self.model_params.emission_mean_function(z, x)
        Cov_Y = lambda z: self.model_params.emission_cov_function(z, x)
        
        C = jnp.atleast_1d(m_Y(m)).shape[0]
        H = _jacrev_2d(m_Y, m)
        if self.model_params.adaptive_emission_cov:
            R = jnp.eye(C) * obs_noise_var
        else:
            R = jnp.atleast_2d(Cov_Y(m))
        
        G = (Lambda**2) / (eta * (eta + Lambda**2))
        V_epi = H @ H.T/eta - (G * (H@U)) @ (H@U).T
        Sigma_obs = V_epi + R
        
        return Sigma_obs
    
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> LoFiBel:
        m, U, Lambda, eta, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.nobs, bel.obs_noise_var
        
        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            _lofi_estimate_noise(m, self.model_params.emission_mean_function,
                                 x, y, nobs, obs_noise_var, 
                                 self.model_params.adaptive_emission_cov)
        
        # Condition on observation.
        m_cond, U_cond, Lambda_cond = \
            _lofi_spherical_cov_condition_on(m, U, Lambda, eta, 
                                             self.model_params.emission_mean_function,
                                             self.model_params.emission_cov_function,
                                             x, y, self.model_params.adaptive_emission_cov,
                                             obs_noise_var_est)
        
        bel_cond = bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
            nobs = nobs_est,
            obs_noise_var = obs_noise_var_est,
        )
        
        return bel_cond
    
    
# Orthogonal LOFI --------------------------------------------------------------
    
class RebayesLoFiOrthogonal(RebayesLoFiSpherical):
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> LoFiBel:
        m, U, Lambda, eta, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.nobs, bel.obs_noise_var
        
        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            _lofi_estimate_noise(m, self.model_params.emission_mean_function,
                                 x, y, nobs, obs_noise_var, 
                                 self.model_params.adaptive_emission_cov)
        
        # Condition on observation.
        m_cond, U_cond, Lambda_cond = \
            _lofi_orth_condition_on(m, U, Lambda, eta, 
                                    self.model_params.emission_mean_function,
                                    self.model_params.emission_cov_function,
                                    x, y, self.model_params.adaptive_emission_cov,
                                    obs_noise_var_est, nobs_est)
        
        bel_cond = bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
            nobs = nobs_est,
            obs_noise_var = obs_noise_var_est,
        )
        
        return bel_cond
    
    
# Diagonal LOFI ----------------------------------------------------------------
    
class RebayesLoFiDiagonal(RebayesLoFi):
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: LoFiBel,
    ) -> LoFiBel:
        m0, m, U, Lambda, eta, gamma, q, Ups = \
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.gamma, bel.q, bel.Ups
        alpha = self.model_params.dynamics_covariance_inflation_factor
        inflation = self.lofi_params.inflation
        
        # Inflate posterior covariance.
        m_infl, U_infl, Lambda_infl, Ups_infl = \
            _lofi_diagonal_cov_inflate(m0, m, U, Lambda, eta, Ups, alpha, inflation)
            
        # Predict dynamics.
        pp_mean_pred, m_pred, U_pred, Lambda_pred, eta_pred, Ups_pred = \
            _lofi_diagonal_cov_predict(m0, m_infl, U_infl, Lambda_infl, gamma, q, eta, Ups_infl)
        
        bel_pred = bel.replace(
            pp_mean = pp_mean_pred,
            mean = m_pred,
            basis = U_pred,
            svs = Lambda_pred,
            eta = eta_pred,
            Ups = Ups_pred,
        )
        
        return bel_pred
    
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, obs_noise_var, Ups = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.obs_noise_var
        m_Y = lambda z: self.model_params.emission_mean_function(z, x)
        Cov_Y = lambda z: self.model_params.emission_cov_function(z, x)
        
        C = jnp.atleast_1d(m_Y(m)).shape[0]
        H = _jacrev_2d(m_Y, m)
        if self.model_params.adaptive_emission_cov:
            R = jnp.eye(C) * obs_noise_var
        else:
            R = jnp.atleast_2d(Cov_Y(m))
        
        P, L = U.shape
        W = U * Lambda
        G = jnp.linalg.pinv(jnp.eye(L) +  W.T @ (W/Ups))
        HW = H/Ups @ W
        V_epi = H @ H.T/Ups - (HW @ G) @ (HW).T
        Sigma_obs = V_epi + R
        
        return Sigma_obs
    
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> LoFiBel:
        m, U, Lambda, Ups, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.Ups, bel.nobs, bel.obs_noise_var
        
        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            _lofi_estimate_noise(m, self.model_params.emission_mean_function,
                                 x, y, nobs, obs_noise_var, 
                                 self.model_params.adaptive_emission_cov)
        
        # Condition on observation.
        m_cond, U_cond, Lambda_cond, Ups_cond = \
            _lofi_diagonal_cov_condition_on(m, U, Lambda, Ups,
                                            self.model_params.emission_mean_function,
                                            self.model_params.emission_cov_function,
                                            x, y, self.model_params.adaptive_emission_cov,
                                            obs_noise_var_est)
        
        bel_cond = bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
            Ups = Ups_cond,
            nobs = nobs_est,
            obs_noise_var = obs_noise_var_est,
        )
        
        return bel_cond
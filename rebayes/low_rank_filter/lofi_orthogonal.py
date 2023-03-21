from functools import partial
from typing import Union, Any

from jax import jit
import jax.numpy as jnp
from jaxtyping import Float, Array

from rebayes.low_rank_filter.lofi import RebayesLoFi, LoFiBel
from rebayes.low_rank_filter.lofi_core import (
    _jacrev_2d,
    _lofi_spherical_cov_inflate,
    _lofi_spherical_cov_predict,
    _lofi_estimate_noise,
    _lofi_orth_condition_on,
)


class RebayesLoFiOrthogonal(RebayesLoFi):
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
    
    # @partial(jit, static_argnums=(0,))
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
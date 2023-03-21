from functools import partial

from jax import jit
from jaxtyping import Float, Array

from rebayes.low_rank_filter.lofi import LoFiBel
from rebayes.low_rank_filter.lofi_spherical import RebayesLoFiSpherical
from rebayes.low_rank_filter.lofi_core import (
    _lofi_estimate_noise,
    _lofi_orth_condition_on,
)


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
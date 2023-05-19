from functools import partial
from typing import Union, Any, Tuple

import chex
import jax
from jax import jit
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jaxtyping import Float, Array
import tensorflow_probability.substrates.jax as tfp

from rebayes.base import (
    FnStateToEmission,
    FnStateToEmission2,
    CovMat,
    EmissionDistFn,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    Rebayes,
)
import rebayes.low_rank_filter.lofi_core as core
from rebayes.utils.sampling import sample_dlr


# Common Classes ---------------------------------------------------------------

tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL

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
    obs_noise_var: float = 1.0


class RebayesLoFi(Rebayes):
    def __init__(
        self,
        dynamics_weights: CovMat,
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        adaptive_emission_cov: bool=False,
        dynamics_covariance_inflation_factor: float=0.0,
        memory_size: int = 10,
        steady_state: bool = False,
        inflation: str = 'bayesian',
        use_svd: bool = True,
    ):
        super().__init__(dynamics_covariance, emission_mean_function, emission_cov_function, emission_dist)
        self.dynamics_weights = dynamics_weights
        self.adaptive_emission_cov = adaptive_emission_cov
        self.dynamics_covariance_inflation_factor = dynamics_covariance_inflation_factor
        self.memory_size = memory_size
        self.steady_state = steady_state
        self.inflation = inflation
        self.use_svd = use_svd

        # Check inflation type
        if inflation not in INFLATION_METHODS:
            raise ValueError(f"Unknown inflation method: {inflation}.")

    def init_bel(
        self, 
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat, 
        X: Float[Array, "input_dim"]=None,
        y: Float[Array, "output_dim"]=None,
    ) -> LoFiBel:
        pp_mean = initial_mean # Predictive prior mean
        init_mean = initial_mean # Initial mean
        memory_size = self.memory_size
        init_basis = jnp.zeros((len(init_mean), memory_size)) # Initial basis
        init_svs = jnp.zeros(memory_size) # Initial singular values
        init_eta = 1 / initial_covariance # Initial precision
        gamma = self.dynamics_weights # Dynamics weights
        q = self.dynamics_covariance # Dynamics covariance
        if self.steady_state: # Steady-state constraint
            q = self.steady_state_constraint(init_eta, gamma)
        init_Ups = jnp.ones((len(init_mean), 1)) * init_eta
        bel = LoFiBel(
            pp_mean=pp_mean,
            mean=init_mean,
            basis=init_basis,
            svs=init_svs,
            eta=init_eta,
            gamma=gamma,
            q=q,
            Ups=init_Ups
        )
        
        return bel

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
        m_Y = lambda z: self.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))

        return y_pred


# Spherical LOFI ---------------------------------------------------------------

class RebayesLoFiSpherical(RebayesLoFi):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: LoFiBel,
    ) -> LoFiBel:
        m0, m, U, Lambda, eta, gamma, q = \
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.gamma, bel.q
        alpha = self.dynamics_covariance_inflation_factor
        inflation = self.inflation

        # Inflate posterior covariance.
        m_infl, U_infl, Lambda_infl, eta_infl = \
            core._lofi_spherical_cov_inflate(m0, m, U, Lambda, eta, alpha, inflation)

        # Predict dynamics.
        predict_fn = core._lofi_spherical_cov_predict if self.use_svd \
            else core._lofi_spherical_cov_svd_free_predict

        pp_mean_pred, m_pred, U_pred, Lambda_pred, eta_pred = \
            predict_fn(m0, m_infl, U_infl, Lambda_infl, gamma, q, eta_infl,
                       self.steady_state)

        bel_pred = bel.replace(
            pp_mean=pp_mean_pred,
            mean=m_pred,
            basis=U_pred,
            svs=Lambda_pred,
            eta=eta_pred,
        )

        return bel_pred

    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, eta = \
            bel.mean, bel.basis, bel.svs, bel.eta
        m_Y = lambda z: self.emission_mean_function(z, x)
        H = core._jacrev_2d(m_Y, m)
        G = (Lambda**2) / (eta * (eta + Lambda**2))
        V_epi = H @ H.T/eta - (G * (H@U)) @ (H@U).T
        R = self.obs_cov(bel, x)
        Sigma_obs = V_epi + R

        return Sigma_obs

    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"]
    ) -> LoFiBel:
        m, U, Lambda, eta, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.nobs, bel.obs_noise_var

        # Condition on observation.
        update_fn = core._lofi_spherical_cov_condition_on if self.use_svd \
            else core._lofi_spherical_cov_svd_free_condition_on

        m_cond, U_cond, Lambda_cond = \
            update_fn(m, U, Lambda, eta, self.emission_mean_function,
                      self.emission_cov_function, x, y,
                      self.adaptive_emission_cov, obs_noise_var)

        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            core._lofi_estimate_noise(m_cond, self.emission_mean_function, x, y, 
                                      nobs, obs_noise_var, self.adaptive_emission_cov)

        bel_cond = bel.replace(
            mean=m_cond,
            basis=U_cond,
            svs=Lambda_cond,
            nobs=nobs_est,
            obs_noise_var=obs_noise_var_est,
        )

        return bel_cond
    
    @partial(jit, static_argnums=(0,))
    def sample_state(
        self,
        bel: LoFiBel, 
        key: Array, 
        n_samples: int=100
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        P, *_ = bel.mean.shape
        diag = bel.eta * jnp.ones((P,))
        shape = (n_samples,)
        params_sample = sample_dlr(key, bel.basis, diag, shape) + bel.mean
        
        return params_sample


# Orthogonal LOFI --------------------------------------------------------------

class RebayesLoFiOrthogonal(RebayesLoFiSpherical):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"]
    ) -> LoFiBel:
        m, U, Lambda, eta, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.nobs, bel.obs_noise_var

        # Condition on observation.
        m_cond, U_cond, Lambda_cond = \
            core._lofi_orth_condition_on(m, U, Lambda, eta, self.emission_mean_function,
                                         self.emission_cov_function, x, y, 
                                         self.adaptive_emission_cov, obs_noise_var, nobs)

        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            core._lofi_estimate_noise(m_cond, self.emission_mean_function, x, y, 
                                      nobs, obs_noise_var, self.adaptive_emission_cov)

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
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: LoFiBel,
    ) -> LoFiBel:
        alpha = self.dynamics_covariance_inflation_factor

        # Inflate posterior covariance.
        inflate_params = core._lofi_diagonal_cov_inflate(
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.Ups, alpha, self.inflation
        )
        m_infl, U_infl, Lambda_infl, Ups_infl = inflate_params

        # Predict dynamics.
        pred_dynamics = core._lofi_diagonal_cov_predict(
            bel.pp_mean, m_infl, U_infl, Lambda_infl, bel.gamma, bel.q, bel.eta, Ups_infl
        )
        pp_mean_pred, m_pred, U_pred, Lambda_pred, eta_pred, Ups_pred = pred_dynamics

        # Update belief 
        bel_pred = bel.replace(
            pp_mean=pp_mean_pred,
            mean=m_pred,
            basis=U_pred,
            svs=Lambda_pred,
            eta=eta_pred,
            Ups=Ups_pred,
        )

        return bel_pred

    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, Ups = \
            bel.mean, bel.basis, bel.svs, bel.Ups
        m_Y = lambda z: self.emission_mean_function(z, x)
        H = core._jacrev_2d(m_Y, m)
        _, L = U.shape
        W = U * Lambda
        G = jnp.linalg.pinv(jnp.eye(L) +  W.T @ (W/Ups))
        HW = H/Ups @ W
        V_epi = H @ H.T/Ups - (HW @ G) @ (HW).T
        R = self.obs_cov(bel, x)
        Sigma_obs = V_epi + R

        return Sigma_obs

    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"]
    ) -> LoFiBel:
        m, U, Lambda, Ups, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.Ups, bel.nobs, bel.obs_noise_var

        # Condition on observation.
        update_fn = core._lofi_diagonal_cov_condition_on if self.use_svd \
            else core._lofi_diagonal_cov_svd_free_condition_on

        m_cond, U_cond, Lambda_cond, Ups_cond = \
            update_fn(m, U, Lambda, Ups, self.emission_mean_function,
                      self.emission_cov_function, x, y,
                      self.adaptive_emission_cov, obs_noise_var)

        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            core._lofi_estimate_noise(m_cond, self.emission_mean_function, x, y, 
                                      nobs, obs_noise_var, self.adaptive_emission_cov)

        bel_cond = bel.replace(
            mean=m_cond,
            basis=U_cond,
            svs=Lambda_cond,
            Ups=Ups_cond,
            nobs=nobs_est,
            obs_noise_var=obs_noise_var_est,
        )

        return bel_cond
    
    @partial(jit, static_argnums=(0,))
    def sample_state(
        self, 
        bel: LoFiBel, 
        key: Array, 
        n_samples: int=100
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        shape = (n_samples,)
        params_sample = sample_dlr(key, bel.basis, bel.Ups.ravel(), shape) + bel.mean
        
        return params_sample

    @partial(jax.jit, static_argnums=(0,4))
    def pred_obs_mc(
        self, 
        bel: LoFiBel,
        key: Array, 
        x: Float[Array, "input_dim"],
        n_samples: int=1,
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        params_sample = self.sample_state(bel, key, n_samples)
        yhat_samples = jax.vmap(self.emission_mean_function, (0, None))(params_sample, x)
        
        return yhat_samples


def init_regression_agent(
        model,
        X_init,
        dynamics_weights,
        dynamics_covariance,
        emission_cov,
        memory_size
):
    key = jax.random.PRNGKey(0)
    _, dim_in = X_init.shape
    pdummy = model.init(key, jnp.ones((1, dim_in)))
    _, recfn = ravel_pytree(pdummy)

    def apply_fn(flat_params, x):
        return model.apply(recfn(flat_params), x)

    agent = RebayesLoFiDiagonal(
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=apply_fn,
        emission_cov_function=lambda m, x: emission_cov,
        adaptive_emission_cov=False,
        dynamics_covariance_inflation_factor=0.0,
        memory_size=memory_size,
        steady_state=False,
        emission_dist=tfd.Normal
    )

    return agent, recfn

from typing import Any, Callable, Literal, Tuple, Union

import chex
from jax import jit
import jax.numpy as jnp
from jaxtyping import Float, Array

from rebayes.base import CovMat
from rebayes.dual_base import (
    DualRebayesParams,
    ObsModel,
    RebayesEstimator,
)
from rebayes.low_rank_filter.lofi_core import (
    _jacrev_2d,
    _lofi_spherical_cov_inflate,
    _lofi_spherical_cov_predict,
    _lofi_spherical_cov_condition_on,
    _lofi_orth_condition_on,
    _lofi_diagonal_cov_inflate,
    _lofi_diagonal_cov_predict,
    _lofi_diagonal_cov_condition_on,
)


# Base Estimator ---------------------------------------------------------------

DualLoFiInflations = Literal["bayesian", "hybrid", "simple"]


@chex.dataclass
class DualLoFiBel:
    pp_mean: chex.Array
    mean: chex.Array
    basis: chex.Array
    svs: chex.Array
    eta: float
    gamma: float
    q: float
    Ups: CovMat = None


@chex.dataclass
class DualLoFiParams:
    memory_size: int
    inflation: DualLoFiInflations
    steady_state: bool = False
    obs_noise_estimator: str = None
    obs_noise_lr_fn: Callable = None


def make_dual_lofi_estimator(
    params: DualRebayesParams,
    obs: ObsModel,
    lofi_params: DualLoFiParams,
):
    def init() -> Tuple[DualRebayesParams, DualLoFiBel]:
        P = params.mu0.shape[0]
        m = lofi_params.memory_size
        
        bel = DualLoFiBel(
            pp_mean = params.mu0,
            mean = params.mu0,
            basis = jnp.zeroes((P, m)),
            svs = jnp.zeros(m),
            eta = params.eta0,
            gamma = params.dynamics_scale_factor,
            q = params.dynamics_noise,
            Ups = jnp.ones((P, 1)) * params.eta0
        )
        
        return params, bel
    
    def predict_state(params, bel):
        return None
    
    def update_state(params, bel, X, Y):
        return None
    
    @jit
    def predict_obs(
        params: DualRebayesParams,
        bel: DualLoFiBel,
        X: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim"], Any]:
        prior_mean = bel.mean
        m_Y = lambda z: obs.emission_mean_function(z, X)
        y_pred = jnp.atleast_1d(m_Y(prior_mean))
        
        return y_pred

    def predict_obs_cov(params, bel, X):
        return None
    
    @jit
    def update_params(
        params: DualRebayesParams,
        t: int,
        X: Float[Array, "input_dim"],
        y: Union[Float[Array, "output_dim"], Any],
        y_pred: Union[Float[Array, "output_dim"], Any],
        bel: DualLoFiBel,
    ) -> DualRebayesParams:
        if lofi_params.obs_noise_estimator is None:
            return params
        
        nobs = params.nobs + 1
        lr = lofi_params.obs_noise_lr_fn(t)
        
        if lofi_params.obs_noise_estimator == "post":
            yhat = jnp.atleast_1d(obs.emission_mean_function(bel.mean, X))
        else:
            yhat = jnp.atleast_1d(y_pred)

        obs_noise = params.obs_noise
        sqerr = ((yhat - y).T @ (yhat - y)).squeeze() / yhat.shape[0]
          
        r = (1-lr)*obs_noise + lr*sqerr
        obs_noise = jnp.max(jnp.array([1e-6, r]))
        params = params.replace(nobs = nobs, obs_noise = obs_noise)
        
        return params
    
    return RebayesEstimator(init, predict_state, update_state, predict_obs, predict_obs_cov, update_params)


# Spherical LOFI Estimator -----------------------------------------------------

def make_dual_lofi_spherical_estimator(
    params: DualRebayesParams,
    obs: ObsModel,
    lofi_params: DualLoFiParams,
):
    @jit
    def predict_state(
        params: DualRebayesParams,
        bel: DualLoFiBel,
    ) -> DualLoFiBel:
        m0, m, U, Lambda, eta, gamma, q = \
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.gamma, bel.q
        alpha = params.cov_inflation_factor
        inflation = lofi_params.inflation
        
        # Inflate posterior covariance.
        m_infl, U_infl, Lambda_infl, eta_infl = \
            _lofi_spherical_cov_inflate(m0, m, U, Lambda, eta, alpha, inflation)
        
        # Predict dynamics.
        pp_mean_pred, m_pred, U_pred, Lambda_pred, eta_pred = \
            _lofi_spherical_cov_predict(m0, m_infl, U_infl, Lambda_infl, gamma, 
                                        q, eta_infl, lofi_params.steady_state)
        
        bel_pred = bel.replace(
            pp_mean = pp_mean_pred,
            mean = m_pred,
            basis = U_pred,
            svs = Lambda_pred,
            eta = eta_pred,
        )
        
        return bel_pred
    
    @jit
    def update_state(
        params: DualRebayesParams,
        bel: DualLoFiBel,
        X: Float[Array, "input_dim"],
        y: Union[Float[Array, "output_dim"], Any],
    ) -> DualLoFiBel:
        m, U, Lambda, eta = bel.mean, bel.basis, bel.svs, bel.eta
        
        # Condition on observation.
        adapt_obs_noise = (lofi_params.obs_noise_estimator is not None)
        m_cond, U_cond, Lambda_cond = \
            _lofi_spherical_cov_condition_on(m, U, Lambda, eta, obs.emission_mean_function,
                                             obs.emission_cov_function, X, y, 
                                             adapt_obs_noise, params.obs_noise)
        
        bel_cond = bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
        )
        
        return bel_cond
    
    @jit
    def predict_obs_cov(
        params: DualRebayesParams,
        bel: DualLoFiBel,
        X: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, eta = bel.mean, bel.basis, bel.svs, bel.eta
        
        m_Y = lambda z: obs.emission_mean_function(z, X)
        Cov_Y = lambda z: obs.emission_cov_function(z, X)
        
        C = jnp.atleast_1d(m_Y(m)).shape[0]
        H = _jacrev_2d(m_Y, m)
        if lofi_params.obs_noise_estimator is not None:
            R = jnp.eye(C) * params.obs_noise
        else:
            R = jnp.atleast_2d(Cov_Y(m))
        
        G = (Lambda**2) / (eta * (eta + Lambda**2))
        V_epi = H @ H.T/eta - (G * (H@U)) @ (H@U).T
        Sigma_obs = V_epi + R
        
        return Sigma_obs
    
    base_estimator = make_dual_lofi_estimator(params, obs, lofi_params)
    estimator = RebayesEstimator(
        init = base_estimator.init,
        predict_state = predict_state,
        update_state = update_state,
        predict_obs = base_estimator.predict_obs,
        predict_obs_cov = predict_obs_cov,
        update_params = base_estimator.update_params
    )
    
    return estimator


# Orthogonal LOFI Estimator ----------------------------------------------------

def make_dual_lofi_orthogonal_estimator(
    params: DualRebayesParams,
    obs: ObsModel,
    lofi_params: DualLoFiParams,
):
    @jit
    def update_state(
        params: DualRebayesParams,
        bel: DualLoFiBel,
        X: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> DualLoFiBel:
        m, U, Lambda, eta = bel.mean, bel.basis, bel.svs, bel.eta
        
        # Condition on observation.
        adapt_obs_noise = (lofi_params.obs_noise_estimator is not None)
        m_cond, U_cond, Lambda_cond = \
            _lofi_orth_condition_on(m, U, Lambda, eta, obs.emission_mean_function,
                                    obs.emission_cov_function, X, y, 
                                    adapt_obs_noise, params.obs_noise, params.nobs)
        
        bel_cond = bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
        )
        
        return bel_cond

    spherical_estimator = make_dual_lofi_spherical_estimator(params, obs, lofi_params)
    estimator = spherical_estimator.replace(update_state = update_state)
    
    return estimator


# Diagonal LOFI Estimator ------------------------------------------------------

def make_dual_lofi_diagonal_estimator(
    params: DualRebayesParams,
    obs: ObsModel,
    lofi_params: DualLoFiParams,
):
    @jit
    def predict_state(
        params: DualRebayesParams,
        bel: DualLoFiBel,
    ) -> DualLoFiBel:
        m0, m, U, Lambda, eta, gamma, q, Ups = \
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.gamma, bel.q, bel.Ups
        alpha = params.cov_inflation_factor
        inflation = lofi_params.inflation
        
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
    
    @jit
    def update_state(
        params: DualRebayesParams,
        bel: DualLoFiBel,
        X: Float[Array, "input_dim"],
        y: Union[Float[Array, "output_dim"], Any],
    ) -> DualLoFiBel:
        m, U, Lambda, Ups = bel.mean, bel.basis, bel.svs, bel.Ups
        
        # Condition on observation.
        adapt_obs_noise = (lofi_params.obs_noise_estimator is not None)
        m_cond, U_cond, Lambda_cond, Ups_cond = \
            _lofi_diagonal_cov_condition_on(m, U, Lambda, Ups, obs.emission_mean_function,
                                            obs.emission_cov_function, X, y, 
                                            adapt_obs_noise, params.obs_noise)
        
        bel_cond = bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
            Ups = Ups_cond,
        )
        
        return bel_cond
    
    @jit
    def predict_obs_cov(
        params: DualRebayesParams,
        bel: DualLoFiBel,
        X: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, Ups = bel.mean, bel.basis, bel.svs, bel.Ups
        
        m_Y = lambda z: obs.emission_mean_function(z, X)
        Cov_Y = lambda z: obs.emission_cov_function(z, X)
        
        C = jnp.atleast_1d(m_Y(m)).shape[0]
        H = _jacrev_2d(m_Y, m)
        if lofi_params.obs_noise_estimator is not None:
            R = jnp.eye(C) * params.obs_noise
        else:
            R = jnp.atleast_2d(Cov_Y(m))
        
        P, L = U.shape
        W = U * Lambda
        G = jnp.linalg.pinv(jnp.eye(L) +  W.T @ (W/Ups))
        HW = H/Ups @ W
        V_epi = H @ H.T/Ups - (HW @ G) @ (HW).T
        Sigma_obs = V_epi + R
        
        return Sigma_obs
    
    base_estimator = make_dual_lofi_estimator(params, obs, lofi_params)
    estimator = RebayesEstimator(
        init = base_estimator.init,
        predict_state = predict_state,
        update_state = update_state,
        predict_obs = base_estimator.predict_obs,
        predict_obs_cov = predict_obs_cov,
        update_params = base_estimator.update_params
    )
    
    return estimator
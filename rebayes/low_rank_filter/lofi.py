from functools import partial

import jax
import chex
from jax import jit
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from typing import Union, Any, Tuple
from jaxtyping import Float, Array
from jax.flatten_util import ravel_pytree
from rebayes.utils.sampling import sample_dlr
from rebayes.base import (
    Belief,
    FnStateToState,
    FnStateToEmission,
    FnStateToEmission2,
    CovMat,
    EmissionDistFn,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    Rebayes,
)

# TODO (proposed): import lofi_core only and access functions using dot notation
from rebayes.low_rank_filter.lofi_core import (
    _jacrev_2d,
    _lofi_spherical_cov_inflate,
    _lofi_spherical_cov_predict,
    _lofi_spherical_cov_svd_free_predict,
    _lofi_estimate_noise,
    _lofi_spherical_cov_condition_on,
    _lofi_spherical_cov_svd_free_condition_on,
    _lofi_orth_condition_on,
    _lofi_diagonal_cov_inflate,
    _lofi_diagonal_cov_predict,
    _lofi_diagonal_cov_condition_on,
    _lofi_diagonal_cov_svd_free_condition_on,
)


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


@chex.dataclass
class LoFiParams:
    """Lightweight container for LOFI parameters.
    """
    initial_mean: Float[Array, "state_dim"]
    initial_covariance: CovMat
    dynamics_weights: CovMat
    dynamics_covariance: CovMat
    emission_mean_function: FnStateAndInputToEmission
    emission_cov_function: FnStateAndInputToEmission2
    emission_dist: EmissionDistFn = \
        lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov))
    adaptive_emission_cov: bool=False
    dynamics_covariance_inflation_factor: float=0.0
    memory_size: int = 10
    steady_state: bool = False
    inflation: str = 'bayesian'
    use_svd: bool = True


class RebayesLoFi(Rebayes):
    def __init__(
        self,
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn,
        dynamics_weights: CovMat,
        memory_size: int,
        # TODO: (deprecate?)
        adaptive_emission_cov: bool,
        dynamics_covariance_inflation_factor: float,
        steady_state: bool,
        inflation: str,
        use_svd: bool,
    ):
        super().__init__(dynamics_covariance, emission_mean_function, emission_cov_function, emission_dist)
        self.memory_size = memory_size
        self.dynamics_weights = dynamics_weights
        self.adaptive_emission_cov = adaptive_emission_cov
        self.dynamics_covariance_inflation_factor = dynamics_covariance_inflation_factor
        self.steady_state = steady_state
        self.inflation = inflation
        self.use_svd = use_svd

        # Check inflation type
        if inflation not in INFLATION_METHODS:
            raise ValueError(f"Unknown inflation method: {inflation}.")

    def init_bel(self, initial_mean, initial_covariance, X, y):
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

        return LoFiBel(
            pp_mean=pp_mean,
            mean=init_mean,
            basis=init_basis,
            svs=init_svs,
            eta=init_eta,
            gamma=gamma,
            q=q,
            Ups=init_Ups
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
        m_Y = lambda z: self.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))

        return y_pred


# Spherical LOFI ---------------------------------------------------------------

class RebayesLoFiSpherical(RebayesLoFi):
    def __init__(
        self,
        params: LoFiParams,
    ):
        super().__init__(params)

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
            _lofi_spherical_cov_inflate(m0, m, U, Lambda, eta, alpha, inflation)

        # Predict dynamics.
        predict_fn = _lofi_spherical_cov_predict if self.use_svd \
            else _lofi_spherical_cov_svd_free_predict

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
        m, U, Lambda, eta, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.obs_noise_var
        m_Y = lambda z: self.emission_mean_function(z, x)
        Cov_Y = lambda z: self.emission_cov_function(z, x)

        C = jnp.atleast_1d(m_Y(m)).shape[0]
        H = _jacrev_2d(m_Y, m)
        if self.adaptive_emission_cov:
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

        # Condition on observation.
        update_fn = _lofi_spherical_cov_condition_on if self.use_svd \
            else _lofi_spherical_cov_svd_free_condition_on

        m_cond, U_cond, Lambda_cond = \
            update_fn(m, U, Lambda, eta, self.emission_mean_function,
                      self.emission_cov_function, x, y,
                      self.adaptive_emission_cov, obs_noise_var)

        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            _lofi_estimate_noise(m_cond, self.emission_mean_function,
                                 x, y, nobs, obs_noise_var, self.adaptive_emission_cov)

        bel_cond = bel.replace(
            mean=m_cond,
            basis=U_cond,
            svs=Lambda_cond,
            nobs=nobs_est,
            obs_noise_var=obs_noise_var_est,
        )

        return bel_cond


# Orthogonal LOFI --------------------------------------------------------------

class RebayesLoFiOrthogonal(RebayesLoFiSpherical):
    def __init__(
        self,
        params: LoFiParams,
    ):
        super().__init__(params)

    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> LoFiBel:
        m, U, Lambda, eta, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.nobs, bel.obs_noise_var

        # Condition on observation.
        m_cond, U_cond, Lambda_cond = \
            _lofi_orth_condition_on(m, U, Lambda, eta,
                                    self.emission_mean_function,
                                    self.emission_cov_function,
                                    x, y, self.adaptive_emission_cov,
                                    obs_noise_var, nobs)

        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            _lofi_estimate_noise(m_cond, self.emission_mean_function,
                                 x, y, nobs, obs_noise_var,
                                 self.adaptive_emission_cov)

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
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn,
        memory_size: int = 10,
        adaptive_emission_cov: bool = False,
        dynamics_covariance_inflation_factor: float = 0.0,
        dynamics_weights: CovMat = 1.0,
        steady_state: bool = False,
        inflation: str = 'bayesian',
        use_svd: bool = True,
    ):
        super().__init__(
        dynamics_covariance, emission_mean_function, emission_cov_function, emission_dist, dynamics_weights,
        memory_size, adaptive_emission_cov, dynamics_covariance_inflation_factor, 
        steady_state, inflation, use_svd
        )

    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: LoFiBel,
    ) -> LoFiBel:
        alpha = self.dynamics_covariance_inflation_factor

        # Inflate posterior covariance.
        inflate_params = _lofi_diagonal_cov_inflate(
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.Ups, alpha, self.inflation
        )
        m_infl, U_infl, Lambda_infl, Ups_infl = inflate_params

        # Predict dynamics.
        pred_dynamics = _lofi_diagonal_cov_predict(bel.pp_mean, m_infl, U_infl, Lambda_infl, bel.gamma, bel.q, bel.eta, Ups_infl)
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
        m, U, Lambda, obs_noise_var, Ups = \
            bel.mean, bel.basis, bel.svs, bel.obs_noise_var, bel.Ups
        m_Y = lambda z: self.emission_mean_function(z, x)
        Cov_Y = lambda z: self.emission_cov_function(z, x)

        C = jnp.atleast_1d(m_Y(m)).shape[0]
        H = _jacrev_2d(m_Y, m)
        if self.adaptive_emission_cov:
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
    def sample_state(self, bel: Belief, key: Array, shape: Tuple) -> Array:
        bel = self.predict_state(bel)
        params_sample = sample_dlr(key, bel.basis, bel.Ups.ravel(), shape) + bel.mean
        return params_sample


    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "obs_dim"]
    ) -> LoFiBel:
        m, U, Lambda, Ups, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.Ups, bel.nobs, bel.obs_noise_var

        # Condition on observation.
        update_fn = _lofi_diagonal_cov_condition_on if self.use_svd \
            else _lofi_diagonal_cov_svd_free_condition_on

        m_cond, U_cond, Lambda_cond, Ups_cond = \
            update_fn(m, U, Lambda, Ups, self.emission_mean_function,
                      self.emission_cov_function, x, y,
                      self.adaptive_emission_cov, obs_noise_var)

        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            _lofi_estimate_noise(m_cond, self.emission_mean_function,
                                 x, y, nobs, obs_noise_var,
                                 self.adaptive_emission_cov)

        bel_cond = bel.replace(
            mean=m_cond,
            basis=U_cond,
            svs=Lambda_cond,
            Ups=Ups_cond,
            nobs=nobs_est,
            obs_noise_var=obs_noise_var_est,
        )

        return bel_cond

    @partial(jax.jit, static_argnums=(0,4))
    def pred_obs_mc(self, key, bel, x, shape=None):
        """
        Sample observations from the posterior predictive distribution.
        """
        shape = shape or (1,)
        # Belief posterior predictive.
        bel = self.predict_state(bel)
        params_sample = sample_dlr(key, bel.basis, bel.Ups.ravel(), shape) + bel.mean
        yhat_samples = jax.vmap(self.emission_mean_function, (0, None))(params_sample, x)
        return yhat_samples

    @partial(jax.jit, static_argnames=("self", "n_samples", "glm_predictive", "clf"))
    def nlpd_mc(self, key, bel, x, y, n_samples=30, glm_predictive=False, clf=False):
        """
        Compute the negative log predictive density (nlpd) as
        a Monte Carlo estimate.
        llfn: log likelihood function
            Takes mean, x, y
        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_1d(y)
        shape = (n_samples,) 
        bel = self.predict_state(bel)
        
        params_sample = sample_dlr(key, bel.basis, bel.Ups.ravel(), shape) + bel.mean
        scale = None if clf else jnp.sqrt(self.emission_cov_function(0.0, 0.0))
        
        def llfn(params, x, y):
            y = y.ravel()
            mean = self.emission_mean_function(params, x).ravel()
            log_likelihood = self.emission_dist(mean, scale).log_prob(y)
            
            return log_likelihood.sum()
        
        def llfn_glm_predictive(params, x, y):
            y = y.ravel()
            m_Y = lambda w: self.emission_mean_function(w, x)
            if clf:
                m_Y = lambda w: jnp.log(self.emission_mean_function(w, x))
            F = _jacrev_2d(m_Y, bel.mean)
            mean = (m_Y(bel.mean) + F @ (params - bel.mean)).ravel()
            if clf:
                mean = jax.nn.softmax(mean)
            log_likelihood = self.emission_dist(mean, scale).log_prob(y)
            
            return log_likelihood.sum()

        # Compute vectorised nlpd
        if glm_predictive:
            llfn = llfn_glm_predictive
            
        vnlpd = lambda w: jax.vmap(llfn, (None, 0, 0))(w, x, y)
        nlpd_vals = -jax.lax.map(vnlpd, params_sample).squeeze()
        nlpd_mean = nlpd_vals.mean(axis=-1)
    
        return nlpd_mean


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

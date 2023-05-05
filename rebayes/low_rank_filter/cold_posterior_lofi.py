from functools import partial
from typing import Any, NamedTuple, Union

import chex
from jax import jit, lax, vmap
import jax.numpy as jnp
from jaxtyping import Array, Float
import tensorflow_probability.substrates.jax as tfp

from rebayes.base import (
    CovMat,
    EmissionDistFn,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    Rebayes,
)
from rebayes.low_rank_filter.lofi_core import (
    _jacrev_2d,
    _lofi_estimate_noise,
    _lofi_diagonal_cov_inflate,
    _lofi_diagonal_cov_predict,
    _lofi_diagonal_cov_condition_on,
    _lofi_diagonal_cov_svd_free_condition_on,
)
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
class ColdPosteriorLoFiBel:
    buffer_X: chex.Array
    buffer_y: chex.Array
    
    pp_mean: chex.Array
    mean: chex.Array
    # mean_lin: chex.Array # Linearization point
    basis: chex.Array
    svs: chex.Array
    eta: float
    gamma: float
    q: float

    Ups: CovMat = None
    nobs: int = 0
    obs_noise_var: float = 1.0
    
    def _update_buffer(self, buffer, item):
        buffer_new = jnp.concatenate([buffer[1:], jnp.expand_dims(item, 0)], axis=0)

        return buffer_new

    def apply_io_buffers(self, X, y):
        buffer_X = self._update_buffer(self.buffer_X, X)
        buffer_y = self._update_buffer(self.buffer_y, y)

        return self.replace(
            buffer_X=buffer_X,
            buffer_y=buffer_y,
        )
    

class ColdPosteriorLoFiParams(NamedTuple):
    buffer_size: int
    dim_input: int
    dim_output: int
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
    n_inner: int = 1


class ColdPosteriorLoFi(Rebayes):
    def __init__(
        self,
        params: ColdPosteriorLoFiParams,
    ):
        super().__init__(params)
        
        # Check inflation type
        if params.inflation not in INFLATION_METHODS:
            raise ValueError(f"Unknown inflation method: {params.inflation}.")

    def init_bel(self):
        pp_mean = self.params.initial_mean # Predictive prior mean
        init_mean = self.params.initial_mean # Initial mean
        memory_size = self.params.memory_size
        init_basis = jnp.zeros((len(init_mean), memory_size)) # Initial basis
        init_svs = jnp.zeros(memory_size) # Initial singular values
        init_eta = 1 / self.params.initial_covariance # Initial precision
        gamma = self.params.dynamics_weights # Dynamics weights
        q = self.params.dynamics_covariance # Dynamics covariance
        if self.params.steady_state: # Steady-state constraint
            q = self.steady_state_constraint(init_eta, gamma)
        init_Ups = jnp.ones((len(init_mean), 1)) * init_eta
        
        # Set up buffers
        L = self.params.buffer_size
        D, C = self.params.dim_input, self.params.dim_output
        if isinstance(D, int):
            buffer_X = jnp.zeros((L, D))
        else:
            buffer_X = jnp.zeros((L, *D))
        buffer_Y = jnp.zeros((L, C))

        return ColdPosteriorLoFiBel(
            buffer_X = buffer_X,
            buffer_y = buffer_Y,
            pp_mean = pp_mean,
            mean = init_mean,
            # mean_lin = init_mean,
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
        bel: ColdPosteriorLoFiBel,
        x: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim"], Any]:
        m = bel.mean
        m_Y = lambda z: self.params.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))

        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_state(
        bel: ColdPosteriorLoFiBel,
    ) -> ColdPosteriorLoFiBel:
        
        raise NotImplementedError
    
    @partial(jit, static_argnums=(0,))
    def _update_state(
        bel: ColdPosteriorLoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> ColdPosteriorLoFiBel:
        
        raise NotImplementedError
    
    @partial(jit, static_argnums=(0,))
    def update_step(
        self,
        bel: ColdPosteriorLoFiBel,
    ) -> ColdPosteriorLoFiBel:
        X, y = bel.buffer_X, bel.buffer_y
        num_timesteps = X.shape[0]
        init_nobs = bel.nobs
        
        def step(t, bel):
            print(t)
            print(f"X[t]: {X[t]}, y[t]: {y[t]}")
            bel_pred = self.predict_state(bel)
            bel_post = self._update_state(bel_pred, X[t], y[t])

            return bel_post
        bel = lax.fori_loop(jnp.maximum(0, num_timesteps-(bel.nobs+1)), num_timesteps, step, bel)
        bel = bel.replace(nobs=init_nobs+1)
        
        return bel
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: ColdPosteriorLoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> ColdPosteriorLoFiBel:
        bel = bel.apply_io_buffers(x, y)
        # bel.mean_lin = bel.mean
        
        def partial_step(_, bel):
            bel = self.update_step(bel)
            
            return bel
        bel = lax.fori_loop(0, self.params.n_inner-1, partial_step, bel)
        bel = self.update_step(bel)
        
        return bel
    

class ColdPosteriorLoFiDiagonal(ColdPosteriorLoFi):
    def __init__(
        self,
        params: ColdPosteriorLoFiParams,
    ):
        super().__init__(params)

    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: ColdPosteriorLoFiBel,
    ) -> ColdPosteriorLoFiBel:
        m0, m, U, Lambda, eta, gamma, q, Ups = \
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.gamma, bel.q, bel.Ups
        alpha = self.params.dynamics_covariance_inflation_factor
        inflation = self.params.inflation

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
        bel: ColdPosteriorLoFiBel,
        x: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, obs_noise_var, Ups = \
            bel.mean, bel.basis, bel.svs, bel.eta, bel.obs_noise_var
        m_Y = lambda z: self.params.emission_mean_function(z, x)
        Cov_Y = lambda z: self.params.emission_cov_function(z, x)

        C = jnp.atleast_1d(m_Y(m)).shape[0]
        H = _jacrev_2d(m_Y, m)
        if self.params.adaptive_emission_cov:
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
    def _update_state(
        self,
        bel: ColdPosteriorLoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> ColdPosteriorLoFiBel:
        m, U, Lambda, Ups, nobs, obs_noise_var = \
            bel.mean, bel.basis, bel.svs, bel.Ups, bel.nobs, bel.obs_noise_var

        # Condition on observation.
        update_fn = _lofi_diagonal_cov_condition_on if self.params.use_svd \
            else _lofi_diagonal_cov_svd_free_condition_on

        m_cond, U_cond, Lambda_cond, Ups_cond = \
            update_fn(m, U, Lambda, Ups, self.params.emission_mean_function,
                      self.params.emission_cov_function, x, y,
                      self.params.adaptive_emission_cov, obs_noise_var)

        # Estimate emission covariance.
        nobs_est, obs_noise_var_est = \
            _lofi_estimate_noise(m_cond, self.params.emission_mean_function,
                                 x, y, nobs, obs_noise_var,
                                 self.params.adaptive_emission_cov)

        bel_cond = bel.replace(
            mean = m_cond,
            basis = U_cond,
            svs = Lambda_cond,
            Ups = Ups_cond,
            nobs = nobs_est,
            obs_noise_var = obs_noise_var_est,
        )

        return bel_cond

    @partial(jit, static_argnums=(0,4))
    def pred_obs_mc(self, key, bel, x, shape=None):
        """
        Sample observations from the posterior predictive distribution.
        """
        shape = shape or (1,)
        # Belief posterior predictive.
        bel = self.predict_state(bel)
        params_sample = sample_dlr(key, bel.basis, bel.Ups.ravel(), shape) + bel.mean
        yhat_samples = vmap(self.params.emission_mean_function, (0, None))(params_sample, x)
        return yhat_samples

    @partial(jit, static_argnames=("self", "n_samples"))
    def nlpd_mc(self, key, bel, x, y, n_samples=30):
        """
        Compute the negative log predictive density (nlpd) as
        a Monte Carlo estimate.
        llfn: log likelihood function
            Takes mean, x, y
        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        shape = (n_samples,)
        bel = self.predict_state(bel)
        params_sample = sample_dlr(key, bel.basis, bel.Ups.ravel(), shape) + bel.mean
        scale = jnp.sqrt(self.params.emission_cov_function(0.0, 0.0))
        def llfn(params, x, y):
            y = y.ravel()
            mean = self.params.emission_mean_function(params, x).ravel()
            log_likelihood = self.params.emission_dist(mean, scale).log_prob(y)
            return log_likelihood.sum()

        # Compute vectorised nlpd
        vnlpd = vmap(llfn, (0, None, None))
        vnlpd = vmap(vnlpd, (None, 0, 0))
        nlpd_vals = -vnlpd(params_sample, x, y).squeeze()

        return nlpd_vals.mean(axis=-1)
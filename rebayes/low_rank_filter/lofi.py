from functools import partial
from typing import Any, Callable, Union

import chex
import jax
from jax import grad, jit
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
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

    Ups: CovMat
    nobs: int
    obs_noise_var: float


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
            Ups=init_Ups,
            nobs=0,
            obs_noise_var=1.0,
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
        super().__init__(dynamics_weights, dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist, adaptive_emission_cov, 
                         dynamics_covariance_inflation_factor, memory_size, steady_state, 
                         inflation, use_svd)

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

    @partial(jit, static_argnums=(0,4))
    def predict_obs_cov(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        aleatoric_factor: float = 1.0,
        apply_fn: Callable = None,
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, eta = \
            bel.mean, bel.basis, bel.svs, bel.eta
        if apply_fn is None:
            m_Y = lambda z: self.emission_mean_function(z, x)
        else:
            m_Y = lambda z: apply_fn(z, x)
        H = core._jacrev_2d(m_Y, m)
        G = (Lambda**2) / (eta * (eta + Lambda**2))
        V_epi = H @ H.T/eta - (G * (H@U)) @ (H@U).T
        R = self.obs_cov(bel, x) * aleatoric_factor
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
        n_samples: int=100,
        temperature: float = 1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        P, *_ = bel.mean.shape
        diag = bel.eta * jnp.ones((P,))
        shape = (n_samples,)
        params_sample = sample_dlr(key, bel.basis, diag, temperature, shape) + \
            bel.mean
        
        return params_sample


# Orthogonal LOFI --------------------------------------------------------------

class RebayesLoFiOrthogonal(RebayesLoFiSpherical):
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
        super().__init__(dynamics_weights, dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist, adaptive_emission_cov, 
                         dynamics_covariance_inflation_factor, memory_size, steady_state, 
                         inflation, use_svd)

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
        super().__init__(dynamics_weights, dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist, adaptive_emission_cov, 
                         dynamics_covariance_inflation_factor, memory_size, steady_state, 
                         inflation, use_svd)

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

    @partial(jit, static_argnums=(0,4))
    def predict_obs_cov(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        aleatoric_factor: float = 1.0,
        apply_fn: Callable = None,
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        m, U, Lambda, Ups = \
            bel.mean, bel.basis, bel.svs, bel.Ups
        if apply_fn is None:
            m_Y = lambda z: self.emission_mean_function(z, x)
        else:
            m_Y = lambda z: apply_fn(z, x)
        H = core._jacrev_2d(m_Y, m)
        _, L = U.shape
        W = U * Lambda
        G = jnp.linalg.pinv(jnp.eye(L) +  W.T @ (W/Ups))
        HW = (H.T/Ups).T @ W
        V_epi = H @ (H.T/Ups) - (HW @ G) @ (HW).T
        R = self.obs_cov(bel, x) * aleatoric_factor
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
    
    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self, 
        bel: LoFiBel, 
        key: Array, 
        n_samples: int=100,
        temperature: float = 1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        shape = (n_samples,)
        params_sample = sample_dlr(key, bel.basis, bel.Ups.ravel(), temperature, shape) + bel.mean
        
        return params_sample

    @partial(jax.jit, static_argnums=(0,4))
    def pred_obs_mc(
        self, 
        bel: LoFiBel,
        key: Array, 
        x: Float[Array, "input_dim"],
        n_samples: int=1,
        temperature: float = 1.0,
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        params_sample = self.sample_state(bel, key, n_samples, temperature)
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
        emission_dist=lambda mean, cov: tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    )

    return agent, recfn


def init_classification_agent(
        model,
        X_init,
        dynamics_weights,
        dynamics_covariance,
        memory_size,
        eps=1e-6,
):
    key = jax.random.PRNGKey(0)
    _, dim_in = X_init.shape
    pdummy = model.init(key, jnp.ones((1, dim_in)))
    _, recfn = ravel_pytree(pdummy)

    def apply_fn(flat_params, x):
        return model.apply(recfn(flat_params), x)

    def emission_cov_fn(flat_params, x):
        p = apply_fn(flat_params, x)
        return jnp.diag(p) - jnp.outer(p, p) + eps * jnp.eye(len(p))

    agent = RebayesLoFiDiagonal(
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=apply_fn,
        emission_cov_function=emission_cov_fn,
        adaptive_emission_cov=False,
        dynamics_covariance_inflation_factor=0.0,
        memory_size=memory_size,
        steady_state=False,
        emission_dist=lambda mean, cov: tfd.Categorical(probs=mean),
    )

    return agent, recfn
    


# Iterated (Diagonal) LOFI -----------------------------------------------------

@chex.dataclass
class IteratedLoFiBel(LoFiBel):
    pp_mean: chex.Array
    mean: chex.Array
    basis: chex.Array
    svs: chex.Array
    eta: float
    gamma: float
    q: float
    S: chex.Array
    T: chex.Array
    
    Ups: CovMat
    nobs: int
    obs_noise_var: float


class RebayesIteratedLoFi(RebayesLoFiDiagonal):
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
        n_replay: int = 10,
        learning_rate: float = 0.01,
    ):
        super().__init__(dynamics_weights, dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist, adaptive_emission_cov, 
                         dynamics_covariance_inflation_factor, memory_size, steady_state, 
                         inflation, use_svd)
        # def log_likelihood(params, x, y):
        #     # Linearized Gaussian observation model
        #     model_mean = self.emission_mean_function(params, x)
        #     cov = self.emission_cov_function(params, x)
        #     linearized_mean = 
        self.log_likelihood = lambda params, x, y: \
            jnp.sum(
                emission_dist(self.emission_mean_function(params, x),
                              self.emission_cov_function(params, x)).log_prob(y)
            )
        self.n_replay = n_replay
        self.learning_rate = learning_rate
        
    def init_bel(
        self, 
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat, 
        X: Float[Array, "input_dim"]=None,
        y: Float[Array, "output_dim"]=None,
    ) -> IteratedLoFiBel:
        init_mean = initial_mean # Initial mean
        memory_size = self.memory_size
        bel = super().init_bel(initial_mean, initial_covariance)
        bel = IteratedLoFiBel(
            **bel,
            S=jnp.zeros((memory_size, len(init_mean))),
            T=jnp.zeros((memory_size, len(init_mean))),
        )
        
        return bel
        
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self,
        bel: IteratedLoFiBel,
    ) -> IteratedLoFiBel:
        alpha = self.dynamics_covariance_inflation_factor

        # Inflate posterior covariance.
        inflate_params = core._lofi_diagonal_cov_inflate(
            bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.Ups, alpha, self.inflation
        )
        m_infl, U_infl, Lambda_infl, Ups_infl = inflate_params

        # Predict dynamics.
        pred_dynamics = core._lofi_iterated_diagonal_cov_predict(
            bel.pp_mean, m_infl, U_infl, Lambda_infl, bel.gamma, bel.q, bel.eta, Ups_infl
        )
        pp_mean_pred, m_pred, U_pred, Lambda_pred, eta_pred, Ups_pred , S_pred, T_pred = \
            pred_dynamics

        # Update belief 
        bel_pred = bel.replace(
            pp_mean=pp_mean_pred,
            mean=m_pred,
            basis=U_pred,
            svs=Lambda_pred,
            eta=eta_pred,
            Ups=Ups_pred,
            S=S_pred,
            T=T_pred,
        )

        return bel_pred
    
    def _update_mean(
        self,
        bel: IteratedLoFiBel,
        m_prev: Float[Array, "state_dim"],
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> IteratedLoFiBel:
        m, Ups, S, T = bel.mean, bel.Ups, bel.S, bel.T
        gll = -jax.grad(self.log_likelihood, argnums=0)(m, x, y)
        additive_term = gll/Ups[:,0] - S.T @ (T @ gll)
        m_cond = m - self.learning_rate * (m - m_prev + additive_term)
        bel_cond = bel.replace(mean=m_cond)
        
        return bel_cond
    
    def _update_precision(
        self,
        bel: IteratedLoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> IteratedLoFiBel:
        bel_cond = super().update_state(bel, x, y)
        bel_cond = bel.replace(
            basis=bel_cond.basis,
            svs=bel_cond.svs,
            Ups=bel_cond.Ups,
            nobs=bel_cond.nobs,
            obs_noise_var=bel_cond.obs_noise_var,
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: IteratedLoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> IteratedLoFiBel:
        m_prev = bel.mean
        def partial_step(_, bel):
            bel = self._update_mean(bel, m_prev, x, y)
            return bel
        bel = jax.lax.fori_loop(0, self.n_replay-1, partial_step, bel)
        bel = self._update_mean(bel, m_prev, x, y)
        bel_cond = self._update_precision(bel, x, y)
        
        return bel_cond
    
    
# Gradient (Diagonal) LOFI -----------------------------------------------------

@chex.dataclass
class GradientLoFiBel:
    pp_mean: chex.Array
    mean: chex.Array
    basis: chex.Array
    svs: chex.Array
    eta: float
    gamma: float
    q: float
    inv_temperature: float
    momentum: chex.Array

    Ups: CovMat
    nobs: int
    obs_noise_var: float

class RebayesGradientLoFi(RebayesLoFiDiagonal):
    def __init__(
        self,
        dynamics_weights: CovMat,
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        loss_fn: Callable = None,
        adaptive_emission_cov: bool=False,
        dynamics_covariance_inflation_factor: float=0.0,
        memory_size: int = 10,
        steady_state: bool = False,
        inflation: str = 'bayesian',
        use_svd: bool = True,
        correction_method: str = "re-sample",
        n_sample: int = 10,
        momentum_weight: float = 0.9,
    ):
        super().__init__(dynamics_weights, dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist, adaptive_emission_cov, 
                         dynamics_covariance_inflation_factor, memory_size, steady_state, 
                         inflation, use_svd)
        self.method = correction_method
        if loss_fn is None:
            self.loss_fn = lambda params, x, y: \
                -jnp.sum(
                    emission_dist(self.emission_mean_function(params, x),
                                self.emission_cov_function(params, x)).log_prob(y)
                )
        else:
            assert self.method == "base"
            self.loss_fn = lambda params, x, y: \
                loss_fn(params, x, y, self.emission_mean_function)
        self.n_sample = n_sample
        self.momentum_weight = momentum_weight
        if self.method not in ["re-sample", "momentum-correction"]:
            raise ValueError("Method must be either 're-sample' or 'momentum-correction'.")
        
    def init_bel(
        self, 
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat, 
        X: Float[Array, "input_dim"]=None,
        y: Float[Array, "output_dim"]=None,
    ) -> GradientLoFiBel:
        state_dim, *_ = initial_mean.shape
        bel = super().init_bel(initial_mean, initial_covariance)
        bel = GradientLoFiBel(
            **bel,
            inv_temperature=0.0, # Start with uniform proposal
            momentum=jnp.zeros((state_dim,)),
        )
        
        return bel
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: GradientLoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> GradientLoFiBel:
        m, U, Lambda, Ups, nobs, obs_noise_var, momentum = \
            bel.mean, bel.basis, bel.svs, bel.Ups, bel.nobs, bel.obs_noise_var, bel.momentum
        key = jr.PRNGKey(nobs)
        
        if self.method == "re-sample":
            m_cond, U_cond, Lambda_cond, Ups_cond = \
                core._lofi_diagonal_gradient_resample_condition_on(
                    m, U, Lambda, Ups, self.emission_mean_function,
                    self.emission_cov_function, x, y, self.emission_dist,
                    self.loss_fn, self.n_sample, key
                )
            momentum_cond = momentum
        elif self.method == "momentum-correction":
            m_cond, U_cond, Lambda_cond, Ups_cond, momentum_cond = \
                core._lofi_diagonal_gradient_momentum_condition_on(
                    m, U, Lambda, Ups, x, y, self.loss_fn,
                    momentum, self.momentum_weight,
                )

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
            momentum=momentum_cond,
        )
        
        return bel_cond
    
    
# OCL (Diagonal) LOFI ----------------------------------------------------------
    
class RebayesOCLLoFiDiagonal(Rebayes):
    def __init__(
        self,
        dynamics_weights: CovMat,
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        adaptive_emission_cov: bool=False,
        decay_dynamics_weight: bool=False,
        dynamics_covariance_inflation_factor: float=0.0,
        memory_size: int = 10,
        steady_state: bool = False,
        inflation: str = 'bayesian',
        use_svd: bool = True,
        learning_rate: float=1e-2,
        gamma_ub: float=None,
    ):
        super().__init__(dynamics_weights, dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist, adaptive_emission_cov, 
                         dynamics_covariance_inflation_factor, memory_size, steady_state, 
                         inflation, use_svd)

        self.delta_ub = None if gamma_ub is None else -2.0 * jnp.log(gamma_ub)
        self.decay_dynamics_weight = decay_dynamics_weight
        self.learning_rate = learning_rate

    @partial(jit, static_argnums=(0,))
    def update_hyperparams_prepred(
        self,
        bel: LoFiBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> LoFiBel:
        alpha = self.dynamics_covariance_inflation_factor
        
        def post_pred_log_likelihood(delta):
            gamma = jnp.exp(-0.5 * delta)
            inflate_params = core._lofi_diagonal_cov_inflate(
                bel.pp_mean, bel.mean, bel.basis, bel.svs, bel.eta, bel.Ups, alpha, self.inflation
            )
            m_infl, U_infl, Lambda_infl, Ups_infl = inflate_params
            pred_dynamics = core._lofi_diagonal_cov_decay_dynamics_predict(
                bel.pp_mean, m_infl, U_infl, Lambda_infl, gamma, bel.q, bel.eta, Ups_infl
            )
            pp_mean_pred, m_pred, U_pred, Lambda_pred, eta_pred, Ups_pred = pred_dynamics
            curr_bel = bel.replace(
                pp_mean=pp_mean_pred,
                mean=m_pred,
                basis=U_pred,
                svs=Lambda_pred,
                eta=eta_pred,
                Ups=Ups_pred,
            )
            y_mean = self.predict_obs(curr_bel, x)
            y_cov = self.predict_obs_cov(curr_bel, x)
            y_dist = self.emission_dist(y_mean, y_cov)
            log_prob = jnp.sum(y_dist.log_prob(y))
            
            return log_prob

        delta = -2 * jnp.log(bel.gamma)
        delta_cond = delta + self.learning_rate * grad(post_pred_log_likelihood)(delta)
        delta_cond = jnp.maximum(delta_cond, 0.0)
        if self.delta_ub is not None:
            delta_cond = jnp.minimum(delta_cond, self.delta_ub)
        gamma_cond = jnp.exp(-0.5 * delta_cond)
        bel_cond = bel.replace(gamma=gamma_cond)
        
        return bel_cond
    
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
        pred_dynamics = core._lofi_diagonal_cov_decay_dynamics_predict(
            bel.pp_mean, m_infl, U_infl, Lambda_infl, bel.gamma, bel.q, bel.eta, 
            Ups_infl, decoupled=not self.decay_dynamics_weight
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


# NF (Diagonal) LOFI ----------------------------------------------------------

# @chex.dataclass
# class NFLoFiBel:
#     pp_mean: chex.Array
#     mean: chex.Array
#     basis: chex.Array
#     svs: chex.Array
#     nf_params: chex.Array
#     eta: float
#     gamma: float
#     q: float

#     Ups: CovMat
#     nobs: int
#     obs_noise_var: float
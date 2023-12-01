from typing import Callable, Union, Tuple, Any

import chex
from functools import partial
import jax
from jax import grad, jit, vmap
from jax.flatten_util import ravel_pytree
from jax.lax import scan
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import tensorflow_probability.substrates.jax as tfp

from rebayes.base import (
    CovMat,
    EmissionDistFn,
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    FnStateToEmission,
    FnStateToEmission2,
    FnStateToState,
    Rebayes,
)
import rebayes.extended_kalman_filter.ekf_core as core
import rebayes.utils.normalizing_flows as nf


tfd = tfp.distributions
MVN = tfd.MultivariateNormalTriL
MVD = tfd.MultivariateNormalDiag


PREDICT_FNS = {
    "fcekf": core._full_covariance_dynamics_predict,
    "fdekf": core._diagonal_dynamics_predict,
    "vdekf": core._diagonal_dynamics_predict,
}

UPDATE_FNS = {
    "fcekf": core._full_covariance_condition_on,
    "fdekf": core._fully_decoupled_ekf_condition_on,
    "vdekf": core._variational_diagonal_ekf_condition_on,
}


# Helper functions
def _process_ekf_cov(cov, P, method):
    if method == "fcekf":
        if isinstance(cov, float):
            cov = cov * jnp.eye(P)
        elif cov.ndim == 1:
            cov = jnp.diag(cov)
    else:
        if isinstance(cov, float):
            cov = cov * jnp.ones(P)

    return cov


@chex.dataclass
class EKFBel:
    mean: chex.Array
    cov: chex.Array
    nobs: int=None
    obs_noise_var: float=None


class RebayesEKF(Rebayes):
    def __init__(
        self,
        dynamics_weights_or_function: Union[float, FnStateToState],
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        adaptive_emission_cov: bool = False,
        dynamics_covariance_inflation_factor: float = 0.0,
        method: str="fcekf",
    ):  
        super().__init__(dynamics_covariance, emission_mean_function, emission_cov_function, emission_dist)
        self.dynamics_weights = dynamics_weights_or_function
        self.adaptive_emission_cov = adaptive_emission_cov
        self.dynamics_covariance_inflation_factor = dynamics_covariance_inflation_factor
        
        self.method = method
        if method == "fcekf":
            if isinstance(self.dynamics_weights, float):
                gamma = self.dynamics_weights
                self.dynamics_weights = lambda x: gamma * x
        elif method in ["fdekf", "vdekf"]:
            assert isinstance(self.dynamics_weights, float) # Dynamics should be a scalar
        else:
            raise ValueError('unknown method ', method)        
        self.pred_fn, self.update_fn = PREDICT_FNS[method], UPDATE_FNS[method]
        self.nobs, self.obs_noise_var = 0, 0.0

    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        Xinit: Float[Array, "input_dim"]=None,
        Yinit: Float[Array, "output_dim"]=None,
    ) -> EKFBel:
        P, *_ = initial_mean.shape
        P0 = _process_ekf_cov(initial_covariance, P, self.method)
        bel = EKFBel(
            mean = initial_mean,
            cov = P0,
            nobs = self.nobs,
            obs_noise_var = self.obs_noise_var,
        )
        if Xinit is not None: # warmup sequentially
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
            
        return bel

    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: EKFBel, 
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim"]:
        m = bel.mean
        m_Y = lambda z: self.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,4))
    def predict_obs_cov(
        self, 
        bel: EKFBel,
        x: Float[Array, "input_dim"],
        aleatoric_factor: float = 1.0,
        apply_fn: Callable = None,
    ) -> Float[Array, "output_dim output_dim"]:
        m, P = bel.mean, bel.cov
        if apply_fn is None:
            m_Y = lambda z: self.emission_mean_function(z, x)
        else:
            m_Y = lambda z: apply_fn(z, x)
        H =  core._jacrev_2d(m_Y, m)
        if self.method == 'fcekf':
            V_epi = H @ P @ H.T
        else:
            V_epi = (P * H) @ H.T
        R = self.obs_cov(bel, x) * aleatoric_factor
        P_obs = V_epi + R
        
        return P_obs
    
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self, 
        bel: EKFBel,
    ) -> EKFBel:
        m, P = bel.mean, bel.cov
        dynamics_covariance = _process_ekf_cov(self.dynamics_covariance, m.shape[0], self.method)
        m_pred, P_pred = self.pred_fn(m, P, self.dynamics_weights, dynamics_covariance, 
                                      self.dynamics_covariance_inflation_factor)
        bel_pred = bel.replace(
            mean = m_pred,
            cov = P_pred,
        )
        return bel_pred

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: EKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> EKFBel:
        m, P, nobs, obs_noise_var = bel.mean, bel.cov, bel.nobs, bel.obs_noise_var
        m_cond, P_cond = self.update_fn(m, P, self.emission_mean_function, 
                                        self.emission_cov_function, x, y, 
                                        1, self.adaptive_emission_cov, obs_noise_var)
        nobs_cond, obs_noise_var_cond = \
            core._ekf_estimate_noise(m_cond, self.emission_mean_function, x, y, 
                                     nobs, obs_noise_var, self.adaptive_emission_cov)
        bel_cond = EKFBel(
            mean = m_cond,
            cov = P_cond,
            nobs = nobs_cond,
            obs_noise_var = obs_noise_var_cond
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self, 
        bel: EKFBel,
        key: Array, 
        n_samples: int=100,
        temperature: float=1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        shape = (n_samples,)
        cooled_cov = bel.cov * temperature
        if self.method != "fcekf":
            mvn = MVD(loc=bel.mean, scale_diag=jnp.sqrt(cooled_cov))
        else:
            mvn = MVN(loc=bel.mean, scale_tril=jnp.linalg.cholesky(cooled_cov))
        params_sample = mvn.sample(seed=key, sample_shape=shape)
        
        return params_sample
    
    @partial(jit, static_argnums=(0,4,))
    def pred_obs_mc(
        self, 
        bel: EKFBel,
        key: Array,
        x: Float[Array, "input_dim"],
        n_samples: int=1
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        params_sample = self.sample_state(bel, key, n_samples)
        yhat_samples = vmap(self.emission_mean_function, (0, None))(params_sample, x)
        
        return yhat_samples


def init_regression_agent(
    model,
    X_init,
    dynamics_weights,
    dynamics_covariance,
    emission_cov,
    method
):
    key = jax.random.PRNGKey(0)
    _, dim_in = X_init.shape
    pdummy = model.init(key, jnp.ones((1, dim_in)))
    _, recfn = ravel_pytree(pdummy)

    def apply_fn(flat_params, x):
        return model.apply(recfn(flat_params), x)

    agent = RebayesEKF(
        dynamics_weights_or_function=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=apply_fn,
        emission_cov_function=lambda m, x: emission_cov,
        adaptive_emission_cov=False,
        dynamics_covariance_inflation_factor=0.0,
        emission_dist=lambda mean, cov: tfd.Normal(loc=mean, scale=jnp.sqrt(cov)),
        method=method,
    )

    return agent, recfn


# For estimating the dynamics weights
# Reference: KF for Online Classification of Non-Stationary Data
# (Titsias et al., 2023)
@chex.dataclass
class OCLEKFBel:
    mean: chex.Array
    cov: chex.Array
    nobs: int=None
    obs_noise_var: float=None
    dynamics_decay_delta: float=None # gamma=exp(-0.5*delta)
    
class RebayesOCLEKF(Rebayes):
    def __init__(
        self,
        dynamics_decay_delta: float, # gamma=exp(-0.5*delta)
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        adaptive_emission_cov: bool = False,
        decay_dynamics_weight: bool = False,
        dynamics_covariance_inflation_factor: float = 0.0,
        method: str="fcekf",
        learning_rate: float=1e-2,
        gamma_ub: float=None,
    ):  
        super().__init__(dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist)
        self.dynamics_decay_delta = dynamics_decay_delta
        self.adaptive_emission_cov = adaptive_emission_cov
        self.dynamics_covariance_inflation_factor = dynamics_covariance_inflation_factor
        self.method = method
        self.learning_rate = learning_rate
        self.delta_ub = None if gamma_ub is None else -2.0 * jnp.log(gamma_ub)
        
        self.pred_fn, self.update_fn = core._decay_dynamics_predict, UPDATE_FNS[method]
        if not decay_dynamics_weight:
            self.pred_fn = partial(self.pred_fn, decoupled=True)
        self.nobs, self.obs_noise_var = 0, 0.0

    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        Xinit: Float[Array, "input_dim"]=None,
        Yinit: Float[Array, "output_dim"]=None,
    ) -> OCLEKFBel:
        P, *_ = initial_mean.shape
        P0 = _process_ekf_cov(initial_covariance, P, self.method)
        bel = OCLEKFBel(
            mean = initial_mean,
            cov = P0,
            nobs = self.nobs,
            obs_noise_var = self.obs_noise_var,
            dynamics_decay_delta = self.dynamics_decay_delta,
        )
        if Xinit is not None: # warmup sequentially
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
            
        return bel
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: OCLEKFBel, 
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim"]:
        m = bel.mean
        m_Y = lambda z: self.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,4))
    def predict_obs_cov(
        self, 
        bel: OCLEKFBel,
        x: Float[Array, "input_dim"],
        aleatoric_factor: float = 1.0,
        apply_fn: Callable = None,
    ) -> Float[Array, "output_dim output_dim"]:
        m, P = bel.mean, bel.cov
        if apply_fn is None:
            m_Y = lambda z: self.emission_mean_function(z, x)
        else:
            m_Y = lambda z: apply_fn(z, x)
        H =  core._jacrev_2d(m_Y, m)
        if self.method == 'fcekf':
            V_epi = H @ P @ H.T
        else:
            V_epi = (P * H) @ H.T
        R = self.obs_cov(bel, x) * aleatoric_factor
        P_obs = V_epi + R
        
        return P_obs
    
    @partial(jit, static_argnums=(0,))
    def update_hyperparams_prepred(
        self,
        bel: OCLEKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> OCLEKFBel:
        m, P, delta, obs_noise_var = \
            bel.mean, bel.cov, bel.dynamics_decay_delta, bel.obs_noise_var
        dynamics_covariance = _process_ekf_cov(
            self.dynamics_covariance, m.shape[0], self.method
        )
        
        def post_pred_log_likelihood(delta):
            dynamics_weights = jnp.exp(-0.5 * delta)
            m_pred, P_pred = self.pred_fn(m, P, dynamics_weights, dynamics_covariance,
                                          self.dynamics_covariance_inflation_factor)
            curr_bel = bel.replace(mean=m_pred, cov=P_pred)
            y_mean = self.predict_obs(curr_bel, x)
            y_cov = self.predict_obs_cov(curr_bel, x)
            y_dist = self.emission_dist(y_mean, y_cov)
            log_prob = jnp.sum(y_dist.log_prob(y))
            
            return log_prob
        delta_cond = delta + self.learning_rate * grad(post_pred_log_likelihood)(delta)
        delta_cond = jnp.maximum(delta_cond, 0.0)
        if self.delta_ub is not None:
            delta_cond = jnp.minimum(delta_cond, self.delta_ub)
        bel_cond = bel.replace(dynamics_decay_delta=delta_cond)
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self, 
        bel: OCLEKFBel,
    ) -> OCLEKFBel:
        m, P, delta = bel.mean, bel.cov, bel.dynamics_decay_delta
        dynamics_covariance = _process_ekf_cov(
            self.dynamics_covariance, m.shape[0], self.method
        )
        dynamics_weights = jnp.exp(-0.5 * delta)
        m_pred, P_pred = self.pred_fn(m, P, dynamics_weights, dynamics_covariance, 
                                      self.dynamics_covariance_inflation_factor)
        bel_pred = bel.replace(
            mean = m_pred,
            cov = P_pred,
        )
        
        return bel_pred

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: OCLEKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> OCLEKFBel:
        m, P, nobs, obs_noise_var = bel.mean, bel.cov, bel.nobs, bel.obs_noise_var
        m_cond, P_cond = self.update_fn(m, P, self.emission_mean_function, 
                                        self.emission_cov_function, x, y, 
                                        1, self.adaptive_emission_cov, obs_noise_var)
        nobs_cond, obs_noise_var_cond = \
            core._ekf_estimate_noise(m_cond, self.emission_mean_function, x, y, 
                                     nobs, obs_noise_var, self.adaptive_emission_cov)
        bel_cond = bel.replace(
            mean = m_cond,
            cov = P_cond,
            nobs = nobs_cond,
            obs_noise_var = obs_noise_var_cond
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self, 
        bel: OCLEKFBel,
        key: Array, 
        n_samples: int=100,
        temperature: float=1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        shape = (n_samples,)
        cooled_cov = bel.cov * temperature
        if self.method != "fcekf":
            mvn = MVD(loc=bel.mean, scale_diag=jnp.sqrt(cooled_cov))
        else:
            mvn = MVN(loc=bel.mean, scale_tril=jnp.linalg.cholesky(cooled_cov))
        params_sample = mvn.sample(seed=key, sample_shape=shape)
        
        return params_sample
    
    @partial(jit, static_argnums=(0,4,))
    def pred_obs_mc(
        self, 
        bel: OCLEKFBel,
        key: Array,
        x: Float[Array, "input_dim"],
        n_samples: int=1
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        params_sample = self.sample_state(bel, key, n_samples)
        yhat_samples = vmap(self.emission_mean_function, (0, None))(params_sample, x)
        
        return yhat_samples
    

# For online training of posterior-refining normalizing flows
@chex.dataclass
class NFEKFBel:
    buffer_X: chex.Array
    buffer_y: chex.Array
    
    mean: chex.Array
    cov: chex.Array
    nf_params: chex.Array # Parameters for normalizing flow
    nobs: int=None
    obs_noise_var: float=None
    
    def _update_buffer(self, step, buffer, item):
        ix_buffer = step % len(buffer)
        buffer = buffer.at[ix_buffer].set(item)
        # buffer_new = jnp.concatenate([buffer[1:], jnp.expand_dims(item, 0)], axis=0)

        return buffer

    def apply_io_buffers(self, X, y):
        n_count = self.nobs
        # print(self.buffer_X.shape, self.buffer_y.shape)
        buffer_X = self._update_buffer(n_count, self.buffer_X, X)
        buffer_y = self._update_buffer(n_count, self.buffer_y, y)

        return self.replace(
            buffer_X=buffer_X,
            buffer_y=buffer_y,
        )
    

class RebayesNFEKF(Rebayes):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dynamics_weights_or_function: Union[float, FnStateToState],
        dynamics_covariance: CovMat,
        emission_mean_function: Union[FnStateToEmission, FnStateAndInputToEmission],
        emission_cov_function: Union[FnStateToEmission2, FnStateAndInputToEmission2],
        nf_initial_params: Float[Array, "nf_state_dim"],
        nf_apply_function: Callable,
        emission_dist: EmissionDistFn = \
            lambda mean, cov: MVN(loc=mean, scale_tril=jnp.linalg.cholesky(cov)),
        adaptive_emission_cov: bool = False,
        dynamics_covariance_inflation_factor: float = 0.0,
        method: str="fcekf",
        learning_rate: float=1e-2,
        buffer_size: int=20,
    ):
        super().__init__(dynamics_covariance, emission_mean_function, 
                         emission_cov_function, emission_dist)
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.nf_apply_function = nf_apply_function
        self.dynamics_weights = dynamics_weights_or_function
        self.adaptive_emission_cov = adaptive_emission_cov
        self.dynamics_covariance_inflation_factor = dynamics_covariance_inflation_factor
        self.method = method
        self.learning_rate = learning_rate
        self.nf_initial_params = nf_initial_params
        self.buffer_size = buffer_size
        
        if method == "fcekf":
            if isinstance(self.dynamics_weights, float):
                gamma = self.dynamics_weights
                self.dynamics_weights = lambda x: gamma * x
            self.base_dist = lambda m, P: MVN(loc=m, scale_tril=jnp.linalg.cholesky(P))
        elif method in ["fdekf", "vdekf"]:
            assert isinstance(self.dynamics_weights, float) # Dynamics should be a scalar
            self.base_dist = lambda m, P: MVD(loc=m, scale_diag=jnp.sqrt(P))
        else:
            raise ValueError('unknown method ', method)        
        
        self.pred_fn, self.update_fn = PREDICT_FNS[method], UPDATE_FNS[method]
        self.nobs, self.obs_noise_var = 0, 0.0
        
    def init_bel(
        self,
        initial_mean: Float[Array, "state_dim"],
        initial_covariance: CovMat,
        Xinit: Float[Array, "input_dim"]=None,
        Yinit: Float[Array, "output_dim"]=None,
    ) -> NFEKFBel:
        P, *_ = initial_mean.shape
        P0 = _process_ekf_cov(initial_covariance, P, self.method)
        
        # Set up buffers
        L = self.buffer_size
        D, C = self.dim_input, self.dim_output
        if isinstance(D, int):
            buffer_X = jnp.zeros((L, D))
        else:
            buffer_X = jnp.zeros((L, *D))
        buffer_Y = jnp.zeros((L, C))
        
        bel = NFEKFBel(
            buffer_X = buffer_X,
            buffer_y = buffer_Y,
            mean = initial_mean,
            cov = P0,
            nf_params = self.nf_initial_params,
            nobs = self.nobs,
            obs_noise_var = self.obs_noise_var,
        )
        if Xinit is not None:
            bel, _ = self.scan(Xinit, Yinit, bel=bel)
        
        return bel
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: NFEKFBel, 
        x: Float[Array, "input_dim"],
    ) -> Float[Array, "output_dim"]:
        m = bel.mean
        bijector = nf.construct_flow(self.nf_apply_function, bel.nf_params)
        # nvp = tfd.TransformedDistribution(
        #     distribution=self.base_dist(m, P),
        #     bijector=bijector
        # )
        m_Y = lambda z: self.emission_mean_function(bijector.forward(z), x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def obs_cov(
        self,
        bel: NFEKFBel,
        X: Float[Array, "input_dim"]
    ) -> Union[Float[Array, "output_dim output_dim"], Any]:
        """Return R(t)"""
        y_pred = jnp.atleast_1d(self.predict_obs(bel, X))
        bijector = nf.construct_flow(self.nf_apply_function, bel.nf_params)
        C, *_ = y_pred.shape
        if self.adaptive_emission_covariance:
            R = bel.obs_noise_var * jnp.eye(C)
        else:
            R = jnp.atleast_2d(self.emission_cov_function(bijector.forward(bel.mean), X))
        
        return R
    
    @partial(jit, static_argnums=(0,4))
    def predict_obs_cov(
        self, 
        bel: NFEKFBel,
        x: Float[Array, "input_dim"],
        aleatoric_factor: float = 1.0,
        apply_fn: Callable = None,
    ) -> Float[Array, "output_dim output_dim"]:
        m, P = bel.mean, bel.cov
        bijector = nf.construct_flow(self.nf_apply_function, bel.nf_params)
        if apply_fn is None:
            m_Y = lambda z: self.emission_mean_function(bijector.forward(z), x)
        else:
            m_Y = lambda z: apply_fn(bijector.forward(z), x)
        H =  core._jacrev_2d(m_Y, m)
        if self.method == 'fcekf':
            V_epi = H @ P @ H.T
        else:
            V_epi = (P * H) @ H.T
        R = self.obs_cov(bel, x) * aleatoric_factor
        P_obs = V_epi + R
        
        return P_obs
    
    @partial(jit, static_argnums=(0,))
    def update_hyperparams_postpred(
        self,
        bel: NFEKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> NFEKFBel:
        bel = bel.apply_io_buffers(x, y)
        X_buffer, y_buffer = bel.buffer_X, bel.buffer_y
        nf_params = bel.nf_params
        
        def post_pred_log_likelihood(nf_params, x, y):
            curr_bel = bel.replace(nf_params=nf_params)
            y_mean = self.predict_obs(curr_bel, x)
            y_cov = self.predict_obs_cov(curr_bel, x)
            y_dist = self.emission_dist(y_mean, y_cov)
            log_prob = jnp.sum(y_dist.log_prob(y))
            
            return log_prob
        post_pred_ll_fn = lambda nf_params: \
            jnp.mean(vmap(post_pred_log_likelihood, (None, 0, 0))(nf_params, X_buffer, y_buffer))
        
        nf_params_cond = nf_params + self.learning_rate * grad(post_pred_ll_fn)(nf_params)
        bel_cond = bel.replace(nf_params=nf_params_cond)
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,))
    def predict_state(
        self, 
        bel: NFEKFBel,
    ) -> NFEKFBel:
        m, P = bel.mean, bel.cov
        dynamics_covariance = _process_ekf_cov(
            self.dynamics_covariance, m.shape[0], self.method
        )
        m_pred, P_pred = self.pred_fn(m, P, self.dynamics_weights, dynamics_covariance, 
                                      self.dynamics_covariance_inflation_factor)
        bel_pred = bel.replace(
            mean = m_pred,
            cov = P_pred,
        )
        
        return bel_pred

    @partial(jit, static_argnums=(0,))
    def update_state(
        self, 
        bel: NFEKFBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> NFEKFBel:
        m, P, nobs, obs_noise_var, nf_params = \
            bel.mean, bel.cov, bel.nobs, bel.obs_noise_var, bel.nf_params
        bijector = nf.construct_flow(self.nf_apply_function, nf_params)
        emission_mean_fn = lambda z, x: self.emission_mean_function(
            bijector.forward(z), x
        )
        emission_cov_fn = lambda z, x: self.emission_cov_function(
            bijector.forward(z), x
        )
        m_cond, P_cond = self.update_fn(m, P, emission_mean_fn, 
                                        emission_cov_fn, x, y, 1, 
                                        self.adaptive_emission_cov, obs_noise_var)
        nobs_cond, obs_noise_var_cond = \
            core._ekf_estimate_noise(m_cond, emission_mean_fn, x, y, 
                                     nobs, obs_noise_var, self.adaptive_emission_cov)
        
        
        bel_cond = bel.replace(
            mean = m_cond,
            cov = P_cond,
            nobs = nobs_cond,
            obs_noise_var = obs_noise_var_cond
        )
        
        return bel_cond
    
    @partial(jit, static_argnums=(0,3))
    def sample_state(
        self, 
        bel: NFEKFBel,
        key: Array, 
        n_samples: int=100,
        temperature: float=1.0,
    ) -> Float[Array, "n_samples state_dim"]:
        bel = self.predict_state(bel)
        shape = (n_samples,)
        cooled_cov = bel.cov * temperature
        if self.method != "fcekf":
            mvn = MVD(loc=bel.mean, scale_diag=jnp.sqrt(cooled_cov))
        else:
            mvn = MVN(loc=bel.mean, scale_tril=jnp.linalg.cholesky(cooled_cov))
        params_sample = mvn.sample(seed=key, sample_shape=shape)
        
        return params_sample
    
    @partial(jit, static_argnums=(0,4,))
    def pred_obs_mc(
        self, 
        bel: NFEKFBel,
        key: Array,
        x: Float[Array, "input_dim"],
        n_samples: int=1
    ) -> Float[Array, "n_samples output_dim"]:
        """
        Sample observations from the posterior predictive distribution.
        """
        params_sample = self.sample_state(bel, key, n_samples)
        bijector = nf.construct_flow(self.nf_apply_function, bel.nf_params)
        emission_mean_fn = lambda z, x: self.emission_mean_function(
            bijector.forward(z), x
        )
        yhat_samples = vmap(emission_mean_fn, (0, None))(params_sample, x)
        
        return yhat_samples
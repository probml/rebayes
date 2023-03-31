import jax
from functools import partial
from jax import jit, lax, jacrev, vmap
from jax import numpy as jnp
from jaxtyping import Float, Array
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiag as MVN
import chex

from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered
from rebayes.base import Rebayes, RebayesParams 
from rebayes.extended_kalman_filter.ekf_core import _stationary_dynamics_diagonal_predict, _full_covariance_dynamics_predict, _diagonal_dynamics_predict
from rebayes.extended_kalman_filter.ekf_core import _full_covariance_condition_on,  _variational_diagonal_ekf_condition_on,  _fully_decoupled_ekf_condition_on



# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))


@chex.dataclass
class EKFBel:
    mean: chex.Array
    cov: chex.Array
    nobs: int=None
    obs_noise_var: float=None


class RebayesEKF(Rebayes):
    def __init__(
        self,
        params: RebayesParams,
        method: str,
    ):
        self.params = params
        self.method = method
        if method not in ['fcekf', 'vdekf', 'fdekf']:
            raise ValueError('unknown method ', method)
        initial_cov = params.initial_covariance
        # assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
        self.eta = 1/initial_cov
        self.gamma = params.dynamics_weights
        # assert isinstance(self.gamma, float), "Dynamics decay term must be a scalar."
        self.q = params.dynamics_covariance
        self.adaptive_variance = params.adaptive_emission_cov
        self.alpha = params.dynamics_covariance_inflation_factor
        self.nobs, self.obs_noise_var = 0, 0.0

    def init_bel(self):
        if self.method == 'fcekf':
            cov = self.params.initial_covariance * jnp.eye(self.params.initial_mean.shape[0])
        else:
            cov = self.params.initial_covariance * jnp.ones(self.params.initial_mean.shape[0])
        return EKFBel(
            mean=self.params.initial_mean, 
            cov=cov,
            nobs=self.nobs,
            obs_noise_var=self.obs_noise_var,
        )

    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, P, nobs, obs_noise_var = bel.mean, bel.cov, bel.nobs, bel.obs_noise_var
        if self.method == 'fcekf':
            pred_mean, pred_cov = _full_covariance_dynamics_predict(m, P, self.q, self.gamma, self.alpha)
        else:
            pred_mean, pred_cov = _diagonal_dynamics_predict(m, P, self.q, self.gamma, self.alpha)
        return EKFBel(mean=pred_mean, cov=pred_cov, nobs=nobs, obs_noise_var=obs_noise_var)

    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        prior_mean, prior_cov, obs_noise_var = bel.mean, bel.cov, bel.obs_noise_var
        m_Y = lambda z: self.params.emission_mean_function(z, u)
        y_pred = jnp.atleast_1d(m_Y(prior_mean))
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(self, bel, u):
        prior_mean, prior_cov, obs_noise_var = bel.mean, bel.cov, bel.obs_noise_var
        m_Y = lambda z: self.params.emission_mean_function(z, u)
        Cov_Y = lambda z: self.params.emission_cov_function(z, u)
        H =  _jacrev_2d(m_Y, prior_mean)
        y_pred = jnp.atleast_1d(m_Y(prior_mean))
        if self.adaptive_variance:
            R = jnp.eye(y_pred.shape[0]) * obs_noise_var
        else:
            R = jnp.atleast_2d(Cov_Y(prior_mean))
        if self.method == 'fcekf':
            V_epi = H @ prior_cov @ H.T
        else:
            V_epi = (prior_cov * H) @ H.T
        Sigma_obs = V_epi + R
        return Sigma_obs

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        if self.method == 'fcekf':
            self.update_fn = _full_covariance_condition_on
        elif self.method == 'vdekf':
            self.update_fn = _variational_diagonal_ekf_condition_on
        elif self.method == 'fdekf':
            self.update_fn = _fully_decoupled_ekf_condition_on
        m, P, nobs, obs_noise_var = bel.mean, bel.cov, bel.nobs, bel.obs_noise_var
        mu, Sigma = self.update_fn(m, P, self.params.emission_mean_function, 
                                   self.params.emission_cov_function, u, y, 
                                   num_iter=1, adaptive_variance=self.adaptive_variance,
                                   obs_noise_var=obs_noise_var)
        nobs, obs_noise_var = _ekf_estimate_noise(mu, self.params.emission_mean_function, 
                                                  u, y, nobs, obs_noise_var,
                                                  adaptive_variance=self.adaptive_variance)
        return EKFBel(mean=mu, cov=Sigma, nobs=nobs, obs_noise_var=obs_noise_var)
    
    @partial(jit, static_argnums=(0,4))
    def pred_obs_mc(self, key, bel, x, shape=None):
        """
        Sample observations from the posterior predictive distribution.
        """
        shape = shape or (1,)
        # Belief posterior predictive.
        bel = self.predict_state(bel)
        if self.method != "fcekf":
            cov = jnp.diagflat(bel.cov)
        else:
            cov = bel.cov
        params_sample = jax.random.multivariate_normal(key, bel.mean, cov, shape)
        yhat_samples = vmap(self.params.emission_mean_function, (0, None))(params_sample, x)
        return yhat_samples




def _ekf_estimate_noise(m, y_cond_mean, u, y, nobs, obs_noise_var, adaptive_variance=False):
    """Estimate observation noise based on empirical residual errors.

    Args:
        m (D_hid,): Prior mean.
        y_cond_mean (Callable): Conditional emission mean function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        nobs (int): Number of observations seen so far.
        obs_noise_var (float): Current estimate of observation noise.
        adaptive_variance (bool): Whether to use adaptive variance.

    Returns:
        nobs (int): Updated number of observations seen so far.
        obs_noise_var (float): Updated estimate of observation noise.
    """
    if not adaptive_variance:
        return 0, 0.0

    m_Y = lambda w: y_cond_mean(w, u)
    yhat = jnp.atleast_1d(m_Y(m))
    
    sqerr = ((yhat - y).T @ (yhat - y)).squeeze() / yhat.shape[0]
    nobs += 1
    obs_noise_var = jnp.max(jnp.array([1e-6, obs_noise_var + 1/nobs * (sqerr - obs_noise_var)]))

    return nobs, obs_noise_var



# DEPRECATED

def stationary_dynamics_fully_decoupled_conditional_moments_gaussian_filter(
    model_params: RebayesParams, 
    emissions: Float[Array, "ntime emission_dim"], 
    num_iter: int=1, 
    inputs: Float[Array, "ntime input_dim"]=None,
    adaptive_variance: bool=False
) -> PosteriorGSSMFiltered:
    """Run a fully decoupled EKF on a stationary dynamics model.

    Args:
        model_params (RebayesParams): Model parameters.
        emissions (T, D_hid): Sequence of emissions.
        num_iter (int, optional): Number of linearizations around posterior for update step.
        inputs (T, D_in, optional): Array of inputs.

    Returns:
        filtered_posterior: GSSMPosterior instance containing,
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """    
    num_timesteps = len(emissions)
    initial_mean, initial_cov = model_params.initial_mean, model_params.initial_covariance
    dynamics_cov = model_params.dynamics_covariance

    # Process conditional emission moments to take in control inputs
    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    m_Y, Cov_Y  = (_process_fn(fn, inputs) for fn in (m_Y, Cov_Y))
    inputs = _process_input(inputs, num_timesteps)
    nobs, obs_noise_var = 0, 0.0

    def _step(carry, t):
        pred_mean, pred_cov_diag, nobs, obs_noise_var = carry

        # Get parameters and inputs for time index t
        Q_diag = _get_params(dynamics_cov, 1, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        filtered_mean, filtered_cov_diag = \
            _fully_decoupled_ekf_condition_on(pred_mean, pred_cov_diag, m_Y, Cov_Y, 
                                              u, y, num_iter, adaptive_variance, obs_noise_var)

        # Update observation noise
        nobs, obs_noise_var = _ekf_estimate_noise(filtered_mean, m_Y, u, y, nobs, obs_noise_var, adaptive_variance)

        # Predict the next state
        pred_mean, pred_cov_diag = _stationary_dynamics_diagonal_predict(filtered_mean, filtered_cov_diag, Q_diag)

        return (pred_mean, pred_cov_diag, nobs, obs_noise_var), (filtered_mean, filtered_cov_diag)

    # Run the general linearization filter
    carry = (initial_mean, initial_cov, nobs, obs_noise_var)
    _, (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=None, filtered_means=filtered_means, filtered_covariances=filtered_covs)


def stationary_dynamics_variational_diagonal_extended_kalman_filter(
    model_params: RebayesParams, 
    emissions: Float[Array, "ntime emission_dim"], 
    num_iter: int=1, 
    inputs: Float[Array, "ntime input_dim"]=None,
    adaptive_variance: bool=False
) -> PosteriorGSSMFiltered:
    """Run a variational diagonal EKF on a stationary dynamics model.

    Args:
        model_params (RebayesParams): Model parameters.
        emissions (T, D_hid): Sequence of emissions.
        num_iter (int, optional): Number of linearizations around posterior for update step.
        inputs (T, D_in, optional): Array of inputs.

    Returns:
        filtered_posterior: GSSMPosterior instance containing,
            filtered_means (T, D_hid)
            filtered_covariances (T, D_hid, D_hid)
    """    
    num_timesteps = len(emissions)
    initial_mean, initial_cov = model_params.initial_mean, model_params.initial_covariance
    dynamics_cov = model_params.dynamics_covariance

    # Process conditional emission moments to take in control inputs
    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    m_Y, Cov_Y  = (_process_fn(fn, inputs) for fn in (m_Y, Cov_Y))
    inputs = _process_input(inputs, num_timesteps)
    nobs, obs_noise_var = 0, 0.0

    def _step(carry, t):
        pred_mean, pred_cov_diag, nobs, obs_noise_var = carry

        # Get parameters and inputs for time index t
        Q_diag = _get_params(dynamics_cov, 1, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        filtered_mean, filtered_cov_diag = \
            _variational_diagonal_ekf_condition_on(pred_mean, pred_cov_diag, m_Y, 
                                                   Cov_Y, u, y, num_iter, adaptive_variance,
                                                   obs_noise_var)

        # Update observation noise
        nobs, obs_noise_var = _ekf_estimate_noise(filtered_mean, m_Y, u, y, nobs, obs_noise_var, adaptive_variance)

        # Predict the next state
        pred_mean, pred_cov_diag = _stationary_dynamics_diagonal_predict(filtered_mean, filtered_cov_diag, Q_diag)

        return (pred_mean, pred_cov_diag, nobs, obs_noise_var), (filtered_mean, filtered_cov_diag)

    # Run the general linearization filter
    carry = (initial_mean, initial_cov, nobs, obs_noise_var)
    _, (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=None, filtered_means=filtered_means, filtered_covariances=filtered_covs)
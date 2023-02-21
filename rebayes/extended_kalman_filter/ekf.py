from functools import partial
from jax import jit, lax, jacrev, vmap
from jax import numpy as jnp
from jaxtyping import Float, Array
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiag as MVN
import chex

from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered
from rebayes.base import _jacrev_2d, Rebayes, RebayesParams, Gaussian


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,)) if x is None else x
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))


@chex.dataclass
class EKFBel:
    mean: chex.Array
    cov: chex.Array
    sse: float=None
    nobs: int=None
    obs_noise_var: float=None


class RebayesEKF(Rebayes):
    def __init__(
        self,
        params: RebayesParams,
        method: str,
        adaptive_variance: bool = False,
        alpha: float=0.0 # Covariance inflation factor for M-EKF
    ):
        self.params = params
        self.method = method
        if method not in ['fcekf', 'vdekf', 'fdekf']:
            raise ValueError('unknown method ', method)
        initial_cov = params.initial_covariance
        assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
        self.eta = 1/initial_cov
        self.gamma = params.dynamics_weights
        assert isinstance(self.gamma, float), "Dynamics decay term must be a scalar."
        self.q = (1 - self.gamma**2) / self.eta
        self.adaptive_variance = adaptive_variance
        self.alpha = alpha
        self.sse, self.nobs, self.obs_noise_var = 0.0, 0, 0.0

    def init_bel(self):
        if self.method == 'fcekf':
            cov = self.params.initial_covariance * jnp.eye(self.params.initial_mean.shape[0])
        else:
            cov = self.params.initial_covariance * jnp.ones(self.params.initial_mean.shape[0])
        return EKFBel(
            mean=self.params.initial_mean, 
            cov=cov,
            sse=self.sse,
            nobs=self.nobs,
            obs_noise_var=self.obs_noise_var,
        )

    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, P, sse, nobs, obs_noise_var = bel.mean, bel.cov, bel.sse, bel.nobs, bel.obs_noise_var
        pred_mean, pred_cov = _non_stationary_dynamics_predict(m, P, self.q, self.gamma)
        return EKFBel(mean=pred_mean, cov=pred_cov, sse=sse, nobs=nobs, obs_noise_var=obs_noise_var)

    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        prior_mean, prior_cov, sse, nobs, obs_noise_var = bel.mean, bel.cov, bel.sse, bel.nobs, bel.obs_noise_var
        m_Y = lambda z: self.params.emission_mean_function(z, u)
        Cov_Y = lambda z: self.params.emission_cov_function(z, u)

        # Predicted mean
        y_pred = jnp.atleast_1d(m_Y(prior_mean))

        # Predicted covariance
        H =  _jacrev_2d(m_Y, prior_mean)
        if self.adaptive_variance:
            R = jnp.eye(y_pred.shape[0]) * obs_noise_var
        else:
            R = jnp.atleast_2d(Cov_Y(prior_mean))
        
        if self.method == 'fcekf':
            V_epi = H @ prior_cov @ H.T
        else:
            V_epi = (prior_cov * H) @ H.T

        Sigma_obs = V_epi + R

        return Gaussian(mean=y_pred, cov=Sigma_obs)

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        if self.method == 'fcekf':
            self.update_fn = _full_covariance_condition_on
        elif self.method == 'vdekf':
            self.update_fn = _variational_diagonal_ekf_condition_on
        elif self.method == 'fdekf':
            self.update_fn = _fully_decoupled_ekf_condition_on
        m, P, sse, nobs, obs_noise_var = bel.mean, bel.cov, bel.sse, bel.nobs, bel.obs_noise_var
        mu, Sigma = self.update_fn(m, P, self.params.emission_mean_function, 
                                   self.params.emission_cov_function, u, y, 
                                   num_iter=1, adaptive_variance=self.adaptive_variance,
                                   obs_noise_var=obs_noise_var, alpha=self.alpha)
        sse, nobs, obs_noise_var = _ekf_estimate_noise(mu, self.params.emission_mean_function, 
                                                       u, y, sse, nobs, obs_noise_var,
                                                       adaptive_variance=self.adaptive_variance)
        return EKFBel(mean=mu, cov=Sigma, sse=sse, nobs=nobs, obs_noise_var=obs_noise_var)


def _full_covariance_condition_on(m, P, y_cond_mean, y_cond_cov, u, y, num_iter, adaptive_variance=False, obs_noise_var=0.0, alpha=0.0):
    """Condition on the emission using a full-covariance EKF.
    Note that this method uses `jnp.linalg.lstsq()` to solve the linear system
    to avoid numerical issues with `jnp.linalg.solve()`.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid, D_hid): Prior covariance.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        num_iter (int): Number of re-linearizations around posterior.
        adaptive_variance (bool): Whether to use adaptive variance.
        alpha (float): Covariance inflation factor.

    Returns:
        mu_cond (D_hid,): Posterior mean.
        Sigma_cond (D_hid, D_hid): Posterior covariance.
    """    
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = jnp.atleast_1d(m_Y(prior_mean))
        if adaptive_variance:
            R = jnp.eye(yhat.shape[0]) * obs_noise_var
        else:
            R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)
        S = R + (H @ prior_cov @ H.T)
        C = prior_cov @ H.T
        K = jnp.linalg.lstsq(S, C.T)[0].T
        posterior_mean = prior_mean + K @ (y - yhat)
        posterior_cov = (1+alpha) * (prior_cov - K @ S @ K.T)
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def _fully_decoupled_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, u, y, num_iter, adaptive_variance=False, obs_noise_var=0.0, alpha=0.0):
    """Condition on the emission using a fully decoupled EKF.

    Args:
        m (D_hid,): Prior mean.
        P_diag (D_hid,): Diagonal elements of prior covariance.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        num_iter (int): Number of re-linearizations around posterior.
        adaptive_variance (bool): Whether to use adaptive variance.
        alpha (float): Covariance inflation factor.

    Returns:
        mu_cond (D_hid,): Posterior mean.
        Sigma_cond (D_hid,): Posterior covariance diagonal elements.
    """    
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = jnp.atleast_1d(m_Y(prior_mean))
        if adaptive_variance:
            R = jnp.eye(yhat.shape[0]) * obs_noise_var
        else:
            R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)
        S = R + (vmap(lambda hh, pp: pp * jnp.outer(hh, hh), (1, 0))(H, prior_cov)).sum(axis=0)
        K = prior_cov[:, None] * jnp.linalg.lstsq(S.T, H)[0].T
        posterior_mean = prior_mean + K @ (y - yhat)
        posterior_cov = (1+alpha) * (prior_cov - prior_cov * vmap(lambda kk, hh: kk @ hh, (0, 1))(K, H))
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P_diag)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def _variational_diagonal_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, u, y, num_iter, adaptive_variance=False, obs_noise_var=0.0, alpha=0.0):
    """Condition on the emission using a variational diagonal EKF.

    Args:
        m (D_hid,): Prior mean.
        P_diag (D_hid,): Diagonal elements of prior covariance.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        num_iter (int): Number of re-linearizations around posterior.
        adaptive_variance (bool): Whether to use adaptive variance.
        alpha (float): Covariance inflation factor.

    Returns:
        mu_cond (D_hid,): Posterior mean.
        Sigma_cond (D_hid,): Posterior covariance diagonal elements.
    """    
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)

    def _step(carry, _):
        prior_mean, prior_cov = carry
        yhat = jnp.atleast_1d(m_Y(prior_mean))
        if adaptive_variance:
            R = jnp.eye(yhat.shape[0]) * obs_noise_var
        else:
            R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)
        K = jnp.linalg.lstsq((R + (prior_cov * H) @ H.T).T, prior_cov * H)[0].T
        R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]
        posterior_cov = 1/(1/prior_cov + ((H.T @ R_inv) * H.T).sum(-1))
        posterior_mean = (1+alpha) * (prior_mean + K @(y - yhat))
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P_diag)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def _stationary_dynamics_diagonal_predict(m, P_diag, Q_diag):
    """Predict the next state using a stationary dynamics model with diagonal covariance matrices.

    Args:
        m (D_hid,): Prior mean.
        P_diag (D_hid,): Diagonal elements of prior covariance.
        Q_diag (D_hid,): Diagonal elements of dynamics covariance.

    Returns:
        mu_pred (D_hid,): Predicted mean.
        Sigma_pred (D_hid,): Predicted covariance diagonal elements.
    """
    mu_pred = m
    Sigma_pred = P_diag + Q_diag
    return mu_pred, Sigma_pred


def _non_stationary_dynamics_predict(m, P, q, gamma):
    """Predict the next state using a non-stationary dynamics model.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid, D_hid): Prior covariance.
        q (float): Dynamics covariance factor.
        gamma (float): Dynamics decay.

    Returns:
        mu_pred (D_hid,): Predicted mean.
        Sigma_pred (D_hid,): Predicted covariance diagonal elements.
    """
    mu_pred = gamma * m
    Sigma_pred = gamma**2 * P
    if P.ndim == 1:
        Sigma_pred += jnp.ones(mu_pred.shape[0]) * q
    else:
        Sigma_pred += jnp.eye(mu_pred.shape[0]) * q
    return mu_pred, Sigma_pred


def _ekf_estimate_noise(m, y_cond_mean, u, y, sse, nobs, obs_noise_var, adaptive_variance=False):
    """Estimate observation noise based on empirical residual errors.

    Args:
        m (D_hid,): Prior mean.
        y_cond_mean (Callable): Conditional emission mean function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        sse (float): Cumulative sum of squared errors.
        nobs (int): Number of observations seen so far.
        obs_noise_var (float): Current estimate of observation noise.
        adaptive_variance (bool): Whether to use adaptive variance.

    Returns:
        sse (float): Updated cumulative sum of squared errors.
        nobs (int): Updated number of observations seen so far.
        obs_noise_var (float): Updated estimate of observation noise.
    """
    if not adaptive_variance:
        return 0.0, 0, 0.0

    m_Y = lambda w: y_cond_mean(w, u)
    yhat = jnp.atleast_1d(m_Y(m))
    
    sqerr = ((yhat - y).T @ (yhat - y)).squeeze() / yhat.shape[0]
    nobs += 1
    obs_noise_var = jnp.max(jnp.array([1e-6, obs_noise_var + 1/nobs * (sqerr - obs_noise_var)]))

    return sse, nobs, obs_noise_var


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
    sse, nobs, obs_noise_var = 0.0, 0, 0.0

    def _step(carry, t):
        pred_mean, pred_cov_diag, sse, nobs, obs_noise_var = carry

        # Get parameters and inputs for time index t
        Q_diag = _get_params(dynamics_cov, 1, t)
        u = inputs[t]
        y = emissions[t]

        # Condition on the emission
        filtered_mean, filtered_cov_diag = \
            _fully_decoupled_ekf_condition_on(pred_mean, pred_cov_diag, m_Y, Cov_Y, 
                                              u, y, num_iter, adaptive_variance, obs_noise_var)

        # Update observation noise
        sse, nobs, obs_noise_var = _ekf_estimate_noise(filtered_mean, m_Y, u, y, sse, nobs, obs_noise_var, adaptive_variance)

        # Predict the next state
        pred_mean, pred_cov_diag = _stationary_dynamics_diagonal_predict(filtered_mean, filtered_cov_diag, Q_diag)

        return (pred_mean, pred_cov_diag, sse, nobs, obs_noise_var), (filtered_mean, filtered_cov_diag)

    # Run the general linearization filter
    carry = (initial_mean, initial_cov, sse, nobs, obs_noise_var)
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
    sse, nobs, obs_noise_var = 0.0, 0, 0.0

    def _step(carry, t):
        pred_mean, pred_cov_diag, sse, nobs, obs_noise_var = carry

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
        sse, nobs, obs_noise_var = _ekf_estimate_noise(filtered_mean, m_Y, u, y, sse, nobs, obs_noise_var, adaptive_variance)

        # Predict the next state
        pred_mean, pred_cov_diag = _stationary_dynamics_diagonal_predict(filtered_mean, filtered_cov_diag, Q_diag)

        return (pred_mean, pred_cov_diag, sse, nobs, obs_noise_var), (filtered_mean, filtered_cov_diag)

    # Run the general linearization filter
    carry = (initial_mean, initial_cov, sse, nobs, obs_noise_var)
    _, (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=None, filtered_means=filtered_means, filtered_covariances=filtered_covs)
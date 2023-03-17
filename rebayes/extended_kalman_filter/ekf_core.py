from functools import partial
from jax import jit, lax, jacrev, vmap
from jax import numpy as jnp
from jaxtyping import Float, Array
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalDiag as MVN
import chex


_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))


def _full_covariance_condition_on(m, P, y_cond_mean, y_cond_cov, u, y, num_iter, adaptive_variance=False, obs_noise_var=0.0):
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
        posterior_cov = prior_cov - K @ S @ K.T
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def _fully_decoupled_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, u, y, num_iter, adaptive_variance=False, obs_noise_var=0.0):
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
        posterior_cov = prior_cov - prior_cov * vmap(lambda kk, hh: kk @ hh, (0, 1))(K, H)
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P_diag)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def _variational_diagonal_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, u, y, num_iter, adaptive_variance=False, obs_noise_var=0.0):
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
        posterior_mean = prior_mean + K @(y - yhat)
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


def _full_covariance_dynamics_predict(m, P, q, gamma, alpha):
    """Predict the next state using a non-stationary dynamics model.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid, D_hid): Prior covariance.
        q (float): Dynamics covariance factor.
        gamma (float): Dynamics decay.
        alpha (float): Covariance inflation factor.

    Returns:
        mu_pred (D_hid,): Predicted mean.
        Sigma_pred (D_hid,): Predicted covariance diagonal elements.
    """
    mu_pred = gamma * m
    Sigma_pred = gamma**2 * P
    Q = jnp.eye(mu_pred.shape[0]) * q
    Sigma_pred += Q
    Sigma_pred += alpha * Sigma_pred # Covariance inflation
    return mu_pred, Sigma_pred


def _diagonal_dynamics_predict(m, P, q, gamma, alpha):
    """Predict the next state using a non-stationary dynamics model.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid, D_hid): Prior covariance.
        q (float): Dynamics covariance factor.
        gamma (float): Dynamics decay.
        alpha (float): Covariance inflation factor.

    Returns:
        mu_pred (D_hid,): Predicted mean.
        Sigma_pred (D_hid,): Predicted covariance diagonal elements.
    """
    mu_pred = gamma * m
    Sigma_pred = gamma**2 * P
    Q = jnp.ones(mu_pred.shape[0]) * q
    Sigma_pred += Q
    Sigma_pred += alpha * Sigma_pred # Covariance inflation
    return mu_pred, Sigma_pred



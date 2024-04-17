from jax import jit, lax, jacrev, vmap
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
MVN = tfd.MultivariateNormalFullCovariance
MVD = tfd.MultivariateNormalDiag


_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))
_exclude_first_timestep = lambda t, A, B: jnp.where(t > 1, A, B)


def _full_covariance_condition_on(m, P, y_cond_mean, y_cond_cov, u, y, num_iter, 
                                  adaptive_variance=False, obs_noise_var=0.0,
                                  inverse_free=False):
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
            R = jnp.atleast_2d(obs_noise_var)
        else:
            R = jnp.atleast_2d(Cov_Y(prior_mean))
        H =  _jacrev_2d(m_Y, prior_mean)
        if inverse_free:
            HTRinv = jnp.linalg.lstsq(R, H)[0].T
            posterior_mean = prior_mean + prior_cov @ HTRinv @ (y - yhat)
            posterior_cov = prior_cov - prior_cov @ HTRinv @ H @ prior_cov
        else:
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


def _fully_decoupled_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, u, y, 
                                      num_iter, adaptive_variance=False, 
                                      obs_noise_var=0.0, inverse_free=False):
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
        if inverse_free:
            raise NotImplementedError
        else:
            S = R + (vmap(lambda hh, pp: pp * jnp.outer(hh, hh), (1, 0))(H, prior_cov)).sum(axis=0)
            K = prior_cov[:, None] * jnp.linalg.lstsq(S.T, H)[0].T
            posterior_mean = prior_mean + K @ (y - yhat)
            posterior_cov = prior_cov - prior_cov * vmap(lambda kk, hh: kk @ hh, (0, 1))(K, H)
        return (posterior_mean, posterior_cov), _

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P_diag)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond


def _variational_diagonal_ekf_condition_on(m, P_diag, y_cond_mean, y_cond_cov, 
                                           u, y, num_iter, adaptive_variance=False, 
                                           obs_noise_var=0.0, inverse_free=False):
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
        if inverse_free:
            HTRinv = jnp.linalg.lstsq(R, H)[0].T
            posterior_mean = prior_mean + (prior_cov * HTRinv.T).T @ (y - yhat)
            posterior_cov = prior_cov - prior_cov**2 * (HTRinv * H.T).sum(-1)
        else:
            K = jnp.linalg.lstsq((R + (prior_cov * H) @ H.T).T, prior_cov * H)[0].T
            R_inv = jnp.linalg.lstsq(R, jnp.eye(R.shape[0]))[0]
            posterior_cov = 1/(1/prior_cov + ((H.T @ R_inv) * H.T).sum(-1))
            posterior_mean = prior_mean + K @ (y - yhat)
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


def _full_covariance_dynamics_predict(m, P, f, Q, alpha):
    """Predict the next state using a non-stationary dynamics model.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid, D_hid): Prior covariance.
        f (Callable): Dynamics function.
        Q (D_hid, D_hid): Dynamics covariance matrix.
        alpha (float): Covariance inflation factor.

    Returns:
        m_pred (D_hid,): Predicted mean.
        P_pred (D_hid, D_hid): Predicted covariance diagonal elements.
    """
    F = _jacrev_2d(f, m)
    m_pred = f(m)
    P_pred = (1 + alpha) * (F @ P @ F.T + Q)
    
    return m_pred, P_pred


def _diagonal_dynamics_predict(m, P, gamma, Q, alpha):
    """Predict the next state using a non-stationary dynamics model.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid,): Prior covariance.
        gamma (float): Dynamics decay.
        Q (D_hid): Diagonal dynamics covariance vector.
        alpha (float): Covariance inflation factor.

    Returns:
        m_pred (D_hid,): Predicted mean.
        P_pred (D_hid,): Predicted covariance diagonal elements.
    """
    m_pred = gamma * m
    P_pred = (1 + alpha) * (gamma**2 * P + Q)
    
    return m_pred, P_pred


def _decay_dynamics_predict(m, P, gamma, Q, alpha, decoupled=False):
    """Predict the next state using a non-stationary dynamics model.

    Args:
        m (D_hid,): Prior mean.
        P (D_hid,): Prior covariance.
        gamma (float): Dynamics decay.
        Q (D_hid): Diagonal dynamics covariance vector.
        alpha (float): Covariance inflation factor.
        decoupled (bool): Whether to use decoupled dynamics.

    Returns:
        m_pred (D_hid,): Predicted mean.
        P_pred (D_hid,): Predicted covariance diagonal elements.
    """
    if decoupled:
        m_pred = m
    else:
        m_pred = gamma * m
    P_pred = (1 + alpha) * (gamma**2 * P + (1 - gamma**2) * Q)
    
    return m_pred, P_pred


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
    nobs += 1
    if not adaptive_variance:
        return nobs, 0.0

    m_Y = lambda w: y_cond_mean(w, u)
    yhat = jnp.atleast_1d(m_Y(m))
    
    sqerr = ((yhat - y).T @ (yhat - y)).squeeze() / yhat.shape[0]
    obs_noise_var = jnp.max(jnp.array([1e-6, obs_noise_var + 1/nobs * (sqerr - obs_noise_var)]))

    return nobs, obs_noise_var


def _swvakf_compute_auxiliary_matrices(f, Q, h, m_prevs, P_prevs, u_buffer, y_buffer, L_eff):
    """Compute auxiliary matrices for the Sliding-Window Adaptive Kalman filter.

    Args:
        f (Callable): Dynamics function.
        Q (D_hid, D_hid,): Dynamics covariance matrix.
        h (Callable): Emission function.
        m_prevs (L, D_hid,): Filtered means buffer.
        P_prevs (L, D_hid, D_hid,): Filtered covariances buffer.
        u_buffer (L, D_in,): Control inputs buffer.
        y_buffer (L, D_obs,): Emissions buffer.
        L_eff (int): Effective lag size.

    Returns:
        A (D_hid, D_hid): Auxiliary matrix A.
        B (D_obs, D_obs): Auxiliary matrix B.
    """    
    L, P, *_ = m_prevs.shape

    def _step(carry, t):
        m_smoothed_next, P_smoothed_next, A_prev, B_prev = carry
        m_filtered, P_filtered = m_prevs[t], P_prevs[t]
        u, y = u_buffer[t], y_buffer[t]
        F = _jacrev_2d(f, m_filtered)
        m_Y = lambda w: h(w, u)
        H = _jacrev_2d(m_Y, m_filtered)

        # Prediction step
        m_pred = f(m_filtered)
        P_pred = F @ P_filtered @ F.T + Q
        G = (jnp.linalg.pinv(P_pred) @ (F @ P_filtered)).T
        # G = jnp.linalg.lstsq(P_pred, F @ P_filtered)[0].T

        # Smoothing step
        m_smoothed = m_filtered + G @ (m_smoothed_next - m_pred)
        P_smoothed = P_filtered + G @ (P_smoothed_next - P_pred) @ G.T
        P_cross = G @ P_smoothed_next
        

        A_inc = P_smoothed_next - (F @ P_cross) - (F @ P_cross).T + F @ P_smoothed @ F.T + \
            jnp.outer(m_smoothed_next - F @ m_smoothed, m_smoothed_next - F @ m_smoothed)
        A_next = jnp.where(L - t > L_eff, A_prev, A_prev + A_inc)
        B_inc = jnp.outer(y - H @ m_smoothed, y - H @ m_smoothed) + H @ P_smoothed @ H.T
        B_next = jnp.where(L - t > L_eff, B_prev, B_prev + B_inc)

        return (m_smoothed, P_smoothed, A_next, B_next), None
    
    # Initial values
    A0 = jnp.zeros((P, P))
    m0, P0 = m_prevs[-1], P_prevs[-1]
    u0, y0 = u_buffer[-1], y_buffer[-1]
    m_Y = lambda w: h(w, u0)
    H0 = _jacrev_2d(m_Y, m_prevs[-1])
    B0 = jnp.outer(y0 - H0 @ m0, y0 - H0 @ m0) + H0 @ P0 @ H0.T
    carry = (m_prevs[-1], P_prevs[-1], A0, B0)

    # Run Kalman smoothing step
    (*_, A, B), _ = lax.scan(_step, carry, jnp.arange(L-2, -1, -1))

    return A, B


def _swvakf_estimate_noise(Q, q_nu, q_psi, R, r_nu, r_psi, A, B, L_eff, rho, t):
    """Estimate observation noise by Bayesian update on Inverse Wishart distributions.

    Args:
        Q (D_hid, D_hid,): Prior dynamics covariance matrix.
        q_nu (float): Prior dynamics covariance matrix degrees of freedom.
        q_psi (D_hid, D_hid,): Prior dynamics covariance matrix scale matrix.
        R (D_obs, D_obs,): Prior emission covariance matrix.
        r_nu (float): Prior emission covariance matrix degrees of freedom.
        r_psi (D_obs, D_obs,): Prior emission covariance matrix scale matrix.
        A (D_hid, D_hid,): Auxiliary matrix A.
        B (D_obs, D_obs,): Auxiliary matrix B.
        L_eff (int): Effective lag size.
        rho (float): Decay factor.
        t (int): Current timestep.

    Returns:
        Q_cond (D_hid, D_hid,): Posterior dynamics covariance matrix.
        q_nu_cond (float): Posterior dynamics covariance matrix degrees of freedom.
        q_psi_cond (D_hid, D_hid,): Posterior dynamics covariance matrix scale matrix.
        R_cond (D_obs, D_obs,): Posterior emission covariance matrix.
        r_nu_cond (float): Posterior emission covariance matrix degrees of freedom.
        r_psi_cond (D_obs, D_obs,): Posterior emission covariance matrix scale matrix.
    """    
    q_nu_pred, q_psi_pred = rho * q_nu, rho * q_psi
    r_nu_pred, r_psi_pred = rho * r_nu, rho * r_psi

    q_nu_cond = _exclude_first_timestep(t, q_nu_pred + L_eff - 1, q_nu)
    q_psi_cond = _exclude_first_timestep(t, q_psi_pred + A, q_psi)
    r_nu_cond = _exclude_first_timestep(t, r_nu_pred + L_eff, r_nu)
    r_psi_cond = _exclude_first_timestep(t, r_psi_pred + B, r_psi)

    Q_cond = _exclude_first_timestep(t, q_psi_cond / q_nu_cond, Q)
    R_cond = _exclude_first_timestep(t, r_psi_cond / r_nu_cond, R)

    return Q_cond, q_nu_cond, q_psi_cond, R_cond, r_nu_cond, r_psi_cond


# Ensemble Kalman filter
def _ensemble_predict(key, ens, gamma, Q):
    """Predict the next ensemble of states.

    Args:
        key (Array): JAX PRNG key.
        ens (D_ens, D_hid,): Prior ensemble of states.
        gamma (float): Dynamics decay.
        Q (D_hid,): Diagonal dynamics covariance matrix.

    Returns:
        ens_pred (D_ens, D_hid,): Predicted ensemble of states.
    """
    *_, D_hid = ens.shape
    ens_pred = gamma * ens
    # Add noise
    ens_noise = MVD(
        loc=jnp.zeros(D_hid,), scale_diag=jnp.sqrt(Q)
    ).sample(seed=key, sample_shape=ens_pred.shape[0])
    ens_pred += ens_noise
    
    return ens_pred


def _ensemble_stochastic_condition_on(
    key, ens, y_cond_mean, y_cond_cov, u, y, 
    adaptive_variance=False, obs_noise_var=0.0
):
    """Condition on the emission using a stochastic ensemble Kalman filter.

    Args:
        key (Array): JAX PRNG key.
        ens (D_ens, D_hid,): Prior ensemble of states.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        u (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        
    Returns:
        ens_cond (D_ens, D_hid,): Posterior ensemble of states.
    """
    m_Y = lambda x: y_cond_mean(x, u)
    Cov_Y = lambda x: y_cond_cov(x, u)
    prior_mean = jnp.mean(ens, axis=0)
    if adaptive_variance:
        R = jnp.atleast_2d(obs_noise_var)
    else:
        R = jnp.atleast_2d(Cov_Y(prior_mean))
    
    # Compute Kalman gain
    L = ens.shape[0] # number of ensemble members
    ens_fwd = vmap(m_Y)(ens).reshape(L, -1)
    ens_fwd_mean = jnp.mean(ens_fwd, axis=0)
    ens_fwd_centered = ens_fwd - ens_fwd_mean
    HPHT = (ens_fwd_centered.T @ ens_fwd_centered) / (L - 1)
    S = HPHT + R
    C = ((ens - prior_mean).T @ ens_fwd_centered) / (L - 1)
    K = jnp.linalg.lstsq(S, C.T)[0].T
    
    # Compute perturbed model observations
    yhat = vmap(m_Y)(ens)
    *_, D_obs = yhat.shape
    yhat_noise = MVN(
        loc=jnp.zeros(D_obs), covariance_matrix=R
    ).sample(seed=key, sample_shape=yhat.shape[0])
    yhat += yhat_noise
    innovations = y - yhat

    update_state = lambda z, e: z + K @ e
    ens_cond = vmap(update_state)(ens, innovations)
    
    return ens_cond
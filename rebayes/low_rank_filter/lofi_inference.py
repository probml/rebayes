"""
Implementation of variants of Orthogonal Recursive Fitting (ORFit) [1] algorithm for online learning.

[1] Min, Y., Ahn, K, & Azizan, N. (2022, July).
One-Pass Learning via Bridging Orthogonal Gradient Descent and Recursive Least-Squares.
Retrieved from https://arxiv.org/abs/2207.13853
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import jacrev, vmap
from jax.lax import scan
from jaxtyping import Float, Array

from rebayes.base import RebayesParams


class LoFiParams(NamedTuple):
    """Lightweight container for ORFit parameters.
    """
    memory_size: int
    sv_threshold: float = 0.0


class PosteriorLoFiFiltered(NamedTuple):
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]
    filtered_sigmas: Float[Array, "ntime memory_size"] = None
    filtered_covariances: Float[Array, "ntime state_dim state_dim"] = None
    filtered_taus: float = None


# Helper functions
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))
_stable_division = lambda a, b: jnp.where(b.any(), a / b, jnp.zeros(shape=a.shape))
_normalize = lambda v: jnp.where(v.any(), v / jnp.linalg.norm(v), jnp.zeros(shape=v.shape))
_projection_matrix = lambda a: _stable_division(a.reshape(-1, 1) @ a.reshape(1, -1), a.T @ a)
_form_projection_matrix = lambda A: jnp.eye(A.shape[0]) - vmap(_projection_matrix, 1)(A).sum(axis=0)
_project = lambda a, x: _stable_division(a * (a.T @ x), (a.T @ a))
_project_to_columns = lambda A, x: \
    jnp.where(A.any(), vmap(_project, (1, None))(A, x).sum(axis=0), jnp.zeros(shape=x.shape))


def _orfit_condition_on(m, U, Sigma, apply_fn, x, y, sv_threshold):
    """Condition on the emission using orfit

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Sigma (D_mem,): Prior singular values.
        apply_fn (Callable): Apply function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        sv_threshold (float): Threshold for singular values.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Sigma_cond (D_mem,): Posterior singular values.
    """    
    f_fn = lambda w: apply_fn(w, x)

    # Compute Jacobians and project out the orthogonal components
    v = jacrev(f_fn)(m).squeeze()
    v_prime = v - _project_to_columns(U, v)

    # Update the U matrix
    u = _normalize(v_prime)
    U_cond = jnp.where(
        Sigma.min() < u @ v_prime, 
        jnp.where(sv_threshold < u @ v_prime, U.at[:, Sigma.argmin()].set(u), U),
        U
    )
    Sigma_cond = jnp.where(
        Sigma.min() < u @ v_prime,
        jnp.where(sv_threshold < u @ v_prime, Sigma.at[Sigma.argmin()].set(u.T @ v_prime), Sigma),
        Sigma,
    )
    # Update the parameters
    m_cond = m - _stable_division((f_fn(m) - y) * v_prime, v.T @ v_prime)

    return m_cond, U_cond, Sigma_cond


def _lofi_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold):
    """Condition step of the low-rank filter algorithm.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Sigma (D_mem,): Prior singular values.
        eta (float): Prior precision.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        sv_threshold (float): Threshold for singular values.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Sigma_cond (D_mem,): Posterior singular values.
    """
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    R = jnp.atleast_2d(Cov_Y(m))
    L = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])
    S = eta*jnp.eye(W_tilde.shape[1]) + W_tilde.T @ W_tilde
    K = (H.T @ A) @ A.T - W_tilde @ (jnp.linalg.pinv(S) @ (W_tilde.T @ ((H.T @ A) @ A.T)))

    m_cond = m + K/eta @ (y - yhat)

    def _update_basis(carry, i):
        U, Sigma = carry
        U_tilde = (H.T - U @ (U.T @ H.T)) @ A
        v = U_tilde[:, i]
        u = _normalize(v)
        U_cond = jnp.where(
            Sigma.min() < u @ v, 
            jnp.where(sv_threshold < u @ v, U.at[:, Sigma.argmin()].set(u), U),
            U
        )
        Sigma_cond = jnp.where(
            Sigma.min() < u @ v,
            jnp.where(sv_threshold < u @ v, Sigma.at[Sigma.argmin()].set(u.T @ v), Sigma),
            Sigma,
        )
        return (U_cond, Sigma_cond), (U_cond, Sigma_cond)

    (U_cond, Sigma_cond), _ = scan(_update_basis, (U, Sigma), jnp.arange(yhat.shape[0]))

    return m_cond, U_cond, Sigma_cond


def _aov_lofi_condition_on(m, U, Sigma, eta, y_cond_mean, x, y, sv_threshold):
    """Condition step of the low-rank filter with adaptive observation variance.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Sigma (D_mem,): Prior singular values.
        eta (float): Prior precision. 
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        sv_threshold (float): Threshold for singular values.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Sigma_cond (D_mem,): Posterior singular values.
        yhat (D_obs,): Emission mean.
    """
    m_Y = lambda w: y_cond_mean(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Sigma * U, (H.T).reshape(U.shape[0], -1)])
    S = eta*jnp.eye(W_tilde.shape[1]) + W_tilde.T @ W_tilde
    K = H.T - W_tilde @ (jnp.linalg.pinv(S) @ (W_tilde.T @ H.T))

    m_cond = m + K/eta @ (y - yhat)

    def _update_basis(carry, i):
        U, Sigma = carry
        U_tilde = H.T - U @ (U.T @ H.T)
        v = U_tilde[:, i]
        u = _normalize(v)
        U_cond = jnp.where(
            Sigma.min() < u @ v, 
            jnp.where(sv_threshold < u @ v, U.at[:, Sigma.argmin()].set(u), U),
            U
        )
        Sigma_cond = jnp.where(
            Sigma.min() < u @ v,
            jnp.where(sv_threshold < u @ v, Sigma.at[Sigma.argmin()].set(u.T @ v), Sigma),
            Sigma,
        )
        return (U_cond, Sigma_cond), (U_cond, Sigma_cond)

    (U_cond, Sigma_cond), _ = scan(_update_basis, (U, Sigma), jnp.arange(yhat.shape[0]))

    return m_cond, U_cond, Sigma_cond


def _aov_lofi_marginalize(m, U, Sigma, eta, y_cond_mean, x, y, nu, rho):
    """Marginalization step of the low-rank filter with adaptive observation variance.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Sigma (D_mem,): Prior singular values.
        eta (float): Prior precision.
        y_cond_mean (Callable): Conditional emission mean function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        nu (float): Student's t-distribution degrees of freedom.
        rho (float): Student's t-distribution scale parameter.

    Returns:
        nu (float): Posterior Student's t-distribution degrees of freedom.
        rho (float): Posterior Student's t-distribution scale parameter.
        tau (float): Posterior mean of precision.
    """
    m_Y = lambda w: y_cond_mean(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Sigma * U, (H.T).reshape(U.shape[0], -1)])
    S = eta*jnp.eye(W_tilde.shape[1]) + W_tilde.T @ W_tilde
    K = H.T - W_tilde @ (jnp.linalg.pinv(S) @ (W_tilde.T @ H.T))

    # Marginalize
    HSHT = 1/eta * H @ K
    nu += yhat.shape[0]
    rho += (y - yhat).T @ (jnp.eye(yhat.shape[0]) + HSHT) @ (y - yhat)
    tau = rho/nu

    return nu, rho, tau


def _lofi_predict(m, Sigma, gamma, q):
    """Predict step of the low-rank filter algorithm.

    Args:
        m (D_hid,): Prior mean.
        Sigma (D_mem,): Prior singluar values.
        gamma (float): Dynamics decay factor.
        q (float): Dynamics noise factor.

    Returns:
        m_pred (D_hid,): Predicted mean.
        Sigma_pred (D_mem,): Predicted singular values.
    """
    m_pred = gamma * m
    Sigma_pred = jnp.sqrt((gamma**2 * Sigma**2)/(jnp.ones(Sigma.shape) + q * Sigma**2))

    return m_pred, Sigma_pred


def orthogonal_recursive_fitting(
    model_params: RebayesParams,
    inf_params: LoFiParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> PosteriorLoFiFiltered:
    """Vectorized implementation of Orthogonal Recursive Fitting (ORFit) algorithm.

    Args:
        model_params (RebayesParams): Model parameters.
        inf_params (LoFiParams): Inference parameters that specify the 
            memory buffer size and singular value threshold.
        emissions (Float[Array]): Array of observations.
        inputs (Float[Array]): Array of inputs.

    Returns:
        filtered_posterior: Posterior object.
    """
    # Initialize parameters
    initial_mean, apply_fn = model_params.initial_mean, model_params.emission_mean_function
    memory_limit, sv_threshold = inf_params
    U, Sigma = jnp.zeros((len(initial_mean), memory_limit)), jnp.zeros((memory_limit,))

    def _step(carry, t):
        params, U, Sigma = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Condition on the emission
        filtered_params, filtered_U, filtered_Sigma = _orfit_condition_on(params, U, Sigma, apply_fn, x, y, sv_threshold)

        return (filtered_params, filtered_U, filtered_Sigma), (filtered_params, filtered_U)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma)
    _, (filtered_means, filtered_bases) = scan(_step, carry, jnp.arange(len(inputs)))
    filtered_posterior =  PosteriorLoFiFiltered(filtered_means=filtered_means, filtered_bases=filtered_bases)

    return filtered_posterior


def low_rank_filter(
    model_params: RebayesParams,
    inf_params: LoFiParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> PosteriorLoFiFiltered:
    """Low-rank filter algorithm.

    Args:
        model_params (RebayesParams): Model parameters.
        inf_params (LoFiParams): Inference parameters that specify the 
            memory buffer size and singular value threshold.
        emissions (Float[Array]): Array of observations.
        inputs (Float[Array]): Array of inputs.

    Returns:
        filtered_posterior: Posterior object.
    """
    # Initialize parameters
    initial_mean, initial_cov = model_params.initial_mean, model_params.initial_covariance
    assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
    eta = 1/initial_cov
    m, sv_threshold = inf_params
    U, Sigma = jnp.zeros((len(initial_mean), m)), jnp.zeros((m,))

    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    gamma = model_params.dynamics_weights
    assert isinstance(gamma, float), "Dynamics decay term must be a scalar."
    
    # Steady-state constraint
    q = (1 - gamma**2) / eta

    def _step(carry, t):
        mean, U, Sigma = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Condition on the emission
        filtered_mean, filtered_U, filtered_Sigma = _lofi_condition_on(mean, U, Sigma, eta, m_Y, Cov_Y, x, y, sv_threshold)

        # Predict the next state
        pred_mean, pred_Sigma = _lofi_predict(filtered_mean, filtered_Sigma, gamma, q)

        return (pred_mean, filtered_U, pred_Sigma), (filtered_mean, filtered_U, filtered_Sigma)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma)
    _, (filtered_means, filtered_bases, filtered_Sigmas) = scan(_step, carry, jnp.arange(len(inputs)))
    filtered_posterior = PosteriorLoFiFiltered(
        filtered_means=filtered_means, 
        filtered_bases=filtered_bases,
        filtered_sigmas=filtered_Sigmas,
    )

    return filtered_posterior


def low_rank_filter_with_adaptive_observation_variance(
    model_params: RebayesParams,
    inf_params: LoFiParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> PosteriorLoFiFiltered:
    """Low-rank filter with adaptive observation variance.

    Args:
        model_params (RebayesParams): Model parameters.
        inf_params (LoFiParams): Inference parameters that specify the 
            memory buffer size and singular value threshold.
        emissions (Float[Array]): Array of observations.
        inputs (Float[Array]): Array of inputs.

    Returns:
        filtered_posterior: Posterior object.
    """
    # Initialize parameters
    initial_mean, initial_cov = model_params.initial_mean, model_params.initial_covariance
    assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
    eta = 1/initial_cov
    m, sv_threshold = inf_params
    U, Sigma = jnp.zeros((len(initial_mean), m)), jnp.zeros((m,))

    m_Y = model_params.emission_mean_function
    gamma = model_params.dynamics_weights
    assert isinstance(gamma, float), "Dynamics decay and noise terms must be scalars."
    nu, rho = 0, 0

    # Steady-state constraint
    q = (1 - gamma**2) / eta

    def _step(carry, t):
        mean, U, Sigma, nu, rho = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Condition on the emission
        filtered_mean, filtered_U, filtered_Sigma = \
            _aov_lofi_condition_on(
                mean, U, Sigma, eta, m_Y, x, y, sv_threshold
            )

        # Predict the next state
        pred_mean, pred_Sigma = _lofi_predict(filtered_mean, filtered_Sigma, gamma, q)

        # Marginalize
        nu, rho, tau = _aov_lofi_marginalize(pred_mean, filtered_U, pred_Sigma, eta, m_Y, x, y, nu, rho)

        return (pred_mean, filtered_U, pred_Sigma, nu, rho), (filtered_mean, filtered_U, filtered_Sigma, tau)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma, nu, rho)
    _, (filtered_means, filtered_bases, filtered_Sigmas, filtered_taus) = scan(_step, carry, jnp.arange(len(inputs)))
    filtered_posterior = PosteriorLoFiFiltered(
        filtered_means=filtered_means, 
        filtered_bases=filtered_bases,
        filtered_sigmas=filtered_Sigmas,
        filtered_taus=filtered_taus,
    )

    return filtered_posterior
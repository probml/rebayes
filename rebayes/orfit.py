"""
Implementation of the Orthogonal Recursive Fitting (ORFit) [1] algorithm for online learning.

[1] Min, Y., Ahn, K, & Azizan, N. (2022, July).
One-Pass Learning via Bridging Orthogonal Gradient Descent and Recursive Least-Squares.
Retrieved from https://arxiv.org/abs/2207.13853
"""

import time
from functools import partial

import jax.numpy as jnp
from jax import jacrev, vmap, jit
from jax.lax import scan
from jaxtyping import Float, Array
from typing import Callable, NamedTuple
import chex
from jax_tqdm import scan_tqdm

from dynamax.nonlinear_gaussian_ssm.models import FnStateAndInputToEmission
from dynamax.generalized_gaussian_ssm.models import FnStateAndInputToEmission2


FnStateInputAndOutputToLoss = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"], Float[Array, "output_dim"]], Float[Array, ""]]


# Helper functions
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))
_stable_division = lambda a, b: jnp.where(b.any(), a / b, jnp.zeros(shape=a.shape))
_normalize = lambda v: jnp.where(v.any(), v / jnp.linalg.norm(v), jnp.zeros(shape=v.shape))
_projection_matrix = lambda a: _stable_division(a.reshape(-1, 1) @ a.reshape(1, -1), a.T @ a)
_form_projection_matrix = lambda A: jnp.eye(A.shape[0]) - vmap(_projection_matrix, 1)(A).sum(axis=0)
_project = lambda a, x: _stable_division(a * (a.T @ x), (a.T @ a))
_project_to_columns = lambda A, x: \
    jnp.where(A.any(), vmap(_project, (1, None))(A, x).sum(axis=0), jnp.zeros(shape=x.shape))


class ORFitParams(NamedTuple):
    """Lightweight container for ORFit parameters.
    """
    initial_mean: Float[Array, "state_dim"]
    apply_function: FnStateAndInputToEmission
    loss_function: FnStateInputAndOutputToLoss
    memory_size: int
    sv_threshold: float = 0.0


class GeneralizedORFitParams(NamedTuple):
    """Lightweight container for ORFit parameters.
    """
    initial_mean: Float[Array, "state_dim"]
    initial_precision: float
    dynamics_decay: float
    dynamics_noise: float
    emission_mean_function: FnStateAndInputToEmission
    emission_cov_function: FnStateAndInputToEmission2
    memory_size: int
    sv_threshold: float = 0.0


class PosteriorORFitFiltered(NamedTuple):
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]


class GeneralizedPosteriorORFitFiltered(NamedTuple):
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]
    filtered_sigmas: Float[Array, "ntime memory_size"]


def _orfit_condition_on(m, U, Sigma, loss_fn, apply_fn, x, y, sv_threshold):
    """Condition on the emission using orfit

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Sigma (D_mem,): Prior singular values.
        loss_fn (Callable): Loss function.
        apply_fn (Callable): Apply function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        sv_threshold (float): Threshold for singular values.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Sigma_cond (D_mem,): Posterior singular values.
    """    
    l_fn = lambda w: loss_fn(w, x, y)
    f_fn = lambda w: apply_fn(w, x)

    # Compute Jacobians and project out the orthogonal components
    g = jacrev(l_fn)(m).squeeze()
    v = jacrev(f_fn)(m).squeeze()
    g_tilde = g - _project_to_columns(U, g)
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
    eta = _stable_division((f_fn(m) - y), (v.T @ g_tilde))
    m_cond = m - eta * g_tilde

    return m_cond, U_cond, Sigma_cond


def _generalized_orfit_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold):
    """Condition step of the ORFit algorithm.

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
    W_tilde = jnp.hstack([Sigma * U, (H.T @ A).squeeze()])
    S = eta*jnp.eye(W_tilde.shape[1]) + W_tilde.T @ W_tilde
    K = (H.T @ A) @ A.T - W_tilde @ (jnp.linalg.pinv(S) @ (W_tilde.T @ ((H.T @ A) @ A.T)))

    m_cond = m + K/eta @ (y - yhat)
    U_tilde = (H.T - U @ (U.T @ H.T)) @ A

    def _update_basis(carry, i):
        U, Sigma = carry
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

    (U_cond, Sigma_cond), _ = scan(_update_basis, (U, Sigma), jnp.arange(U_tilde.shape[1]))

    return m_cond, U_cond, Sigma_cond


def _generalized_orfit_predict(m, Sigma, gamma, q):
    """Predict step of the ORFit algorithm.

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
    model_params: ORFitParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> PosteriorORFitFiltered:
    """Vectorized implementation of Orthogonal Recursive Fitting (ORFit) algorithm.

    Args:
        model_params (ORFitParams): Model parameters.
        emissions (Float[Array]): Array of observations.
        inputs (Float[Array]): Array of inputs.

    Returns:
        filtered_posterior: Posterior object.
    """
    # Initialize parameters
    initial_mean, apply_fn, loss_fn, memory_limit, sv_threshold = model_params
    U, Sigma = jnp.zeros((len(initial_mean), memory_limit)), jnp.zeros((memory_limit,))

    def _step(carry, t):
        params, U, Sigma = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Condition on the emission
        filtered_params, filtered_U, filtered_Sigma = _orfit_condition_on(params, U, Sigma, loss_fn, apply_fn, x, y, sv_threshold)

        return (filtered_params, filtered_U, filtered_Sigma), (filtered_params, filtered_U)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma)
    _, (filtered_means, filtered_bases) = scan(_step, carry, jnp.arange(len(inputs)))
    filtered_posterior =  PosteriorORFitFiltered(filtered_means=filtered_means, filtered_bases=filtered_bases)

    return filtered_posterior


def generalized_orthogonal_recursive_fitting(
    model_params: GeneralizedORFitParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> GeneralizedPosteriorORFitFiltered:
    """Generalized orthogonal recursive fitting algorithm.

    Args:
        model_params (GeneralizedORFitParams): Model parameters.
        emissions (Float[Array]): Array of observations.
        inputs (Float[Array]): Array of inputs.

    Returns:
        filtered_posterior: Posterior object.
    """
    # Initialize parameters
    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    gamma, q, m = model_params.dynamics_decay, model_params.dynamics_noise, model_params.memory_size
    sv_threshold = model_params.sv_threshold
    initial_mean, eta = model_params.initial_mean, model_params.initial_precision
    U, Sigma = jnp.zeros((len(initial_mean), m)), jnp.zeros((m,))

    def _step(carry, t):
        mean, U, Sigma = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Condition on the emission
        filtered_mean, filtered_U, filtered_Sigma = _generalized_orfit_condition_on(mean, U, Sigma, eta, m_Y, Cov_Y, x, y, sv_threshold)

        # Predict the next state
        pred_mean, pred_Sigma = _generalized_orfit_predict(filtered_mean, filtered_Sigma, gamma, q)

        return (pred_mean, filtered_U, pred_Sigma), (filtered_mean, filtered_U, filtered_Sigma)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma)
    _, (filtered_means, filtered_bases, filtered_Sigmas) = scan(_step, carry, jnp.arange(len(inputs)))
    filtered_posterior = GeneralizedPosteriorORFitFiltered(
        filtered_means=filtered_means, 
        filtered_bases=filtered_bases,
        filtered_sigmas=filtered_Sigmas,
    )

    return filtered_posterior


@chex.dataclass
class ORFitBel:
    mean: chex.Array
    basis: chex.Array
    sigma: chex.Array


class RebayesORFit:
    def __init__(
        self,
        orfit_params: ORFitParams,
        method: str,
    ):
        if method == 'orfit':
            self.method = method
            self.update_fn = _orfit_condition_on
            self.apply_fn = orfit_params.apply_function
            self.loss_fn = orfit_params.loss_function
        elif method == 'generalized_orfit':
            self.method = method
            self.eta = orfit_params.initial_precision
            self.update_fn = _generalized_orfit_condition_on
            self.gamma = orfit_params.dynamics_decay
            self.q = orfit_params.dynamics_noise
            self.f = orfit_params.emission_mean_function
            self.r = orfit_params.emission_cov_function
        else:
            raise ValueError(f"Unknown method {method}.")
        self.mu0 = orfit_params.initial_mean
        self.m = orfit_params.memory_size
        self.sv_threshold = orfit_params.sv_threshold
        self.U0 = jnp.zeros((len(self.mu0), self.m))
        self.Sigma0 = jnp.zeros((self.m,))

    def initialize(self):
        return ORFitBel(mean=self.mu0, basis=self.U0, sigma=self.Sigma0)

    @partial(jit, static_argnums=(0,))
    def update(self, bel, u, y):
        m, U, Sigma = bel.mean, bel.basis, bel.sigma # prior predictive for hidden state
        if self.method == 'orfit':
            m_cond, U_cond, Sigma_cond = self.update_fn(m, U, Sigma, self.loss_fn, self.apply_fn, u, y, self.sv_threshold)    
        elif self.method == 'generalized_orfit':
            m_cond, U_cond, Sigma_cond = self.update_fn(m, U, Sigma, self.eta, self.f, self.r, u, y, self.sv_threshold)
        return ORFitBel(mean=m_cond, basis=U_cond, sigma=Sigma_cond)

    def scan(self, X, Y, callback=None):
        num_timesteps = X.shape[0]
        
        @scan_tqdm(num_timesteps)
        def step(bel, t):
            bel = self.update(bel, X[t], Y[t])
            out = None
            if callback is not None:
                out = callback(bel, t, X[t], Y[t])
            return bel, out

        carry = self.initialize()
        bel, outputs = scan(step, carry, jnp.arange(num_timesteps))
        return bel, outputs

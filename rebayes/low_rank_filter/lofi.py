from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import jit, jacrev, vmap
from jax.lax import scan
from jaxtyping import Float, Array
import chex

from rebayes.base import RebayesParams, Rebayes, Gaussian


# Helper functions
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))
_stable_division = lambda a, b: jnp.where(b.any(), a / b, jnp.zeros(shape=a.shape))
_normalize = lambda v: jnp.where(v.any(), v / jnp.linalg.norm(v), jnp.zeros(shape=v.shape))
_projection_matrix = lambda a: _stable_division(a.reshape(-1, 1) @ a.reshape(1, -1), a.T @ a)
_form_projection_matrix = lambda A: jnp.eye(A.shape[0]) - vmap(_projection_matrix, 1)(A).sum(axis=0)
_project = lambda a, x: _stable_division(a * (a.T @ x), (a.T @ a))
_project_to_columns = lambda A, x: \
    jnp.where(A.any(), vmap(_project, (1, None))(A, x).sum(axis=0), jnp.zeros(shape=x.shape))


@chex.dataclass
class LoFiBel:
    mean: chex.Array
    basis: chex.Array
    sigma: chex.Array
    nu: float = None
    rho: float = None


class LoFiParams(NamedTuple):
    """Lightweight container for ORFit parameters.
    """
    memory_size: int
    sv_threshold: float = 0.0
    adaptive_variance: bool = False


class PosteriorLoFiFiltered(NamedTuple):
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]
    filtered_sigmas: Float[Array, "ntime memory_size"] = None
    filtered_covariances: Float[Array, "ntime state_dim state_dim"] = None
    filtered_taus: float = None


class RebayesLoFi(Rebayes):
    def __init__(
        self,
        model_params: RebayesParams,
        orfit_params: LoFiParams,
        method: str,
    ):
        if method == 'orfit':
            pass
        elif method == 'full_svd_lofi' or method == 'orth_svd_lofi':
            initial_cov = model_params.initial_covariance
            assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
            self.eta = 1/initial_cov
            self.gamma = model_params.dynamics_weights
            assert isinstance(self.gamma, float), "Dynamics decay term must be a scalar."
            self.q = (1 - self.gamma**2) / self.eta
        else:
            raise ValueError(f"Unknown method {method}.")
        self.method = method
        self.nu, self.rho = 0.0, 0.0
        self.model_params = model_params
        self.m, self.sv_threshold, self.adaptive_variance = orfit_params
        self.U0 = jnp.zeros((len(model_params.initial_mean), self.m))
        self.Sigma0 = jnp.zeros((self.m,))

    def init_bel(self):
        return LoFiBel(
            mean=self.model_params.initial_mean, basis=self.U0, sigma=self.Sigma0, nu=self.nu, rho=self.rho
        )
    
    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, U, Sigma, nu, rho = bel.mean, bel.basis, bel.sigma, bel.nu, bel.rho
        if self.method == 'orfit':
            return bel
        else:
            m_pred, Sigma_pred = _lofi_predict(m, Sigma, self.gamma, self.q)
            U_pred = U

        return LoFiBel(mean=m_pred, basis=U_pred, sigma=Sigma_pred, nu=nu, rho=rho)

    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        m, U, sigma, nu, rho = bel.mean, bel.basis, bel.sigma, bel.nu, bel.rho
        m_Y = lambda z: self.model_params.emission_mean_function(z, u)
        Cov_Y = lambda z: self.model_params.emission_cov_function(z, u)
        
        # Predicted mean
        y_pred = jnp.atleast_1d(m_Y(m))

        # Predicted covariance
        H =  _jacrev_2d(m_Y, m)
        if self.method == 'orfit':
            Sigma_obs = H @ H.T - (H @ U) @ (H @ U).T
        else:
            if self.adaptive_variance:
                R = jnp.eye(y_pred.shape[0])
            else:
                R = jnp.atleast_2d(Cov_Y(m))
            tau = jnp.where(jnp.isfinite(jnp.divide(rho, nu)), jnp.divide(rho, nu), 1.0)
            D = (sigma**2)/(self.eta**2 * jnp.ones(sigma.shape) + self.eta * sigma**2)
            HU = H @ U
            V_epi = H @ H.T/self.eta - (D * HU) @ (HU).T
            Sigma_obs = tau * V_epi + tau * R
        
        return Gaussian(mean=y_pred, cov=Sigma_obs) # TODO: non-Gaussian distribution support

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        m, U, Sigma, nu, rho = bel.mean, bel.basis, bel.sigma, bel.nu, bel.rho
        if self.method == 'orfit':
            m_cond, U_cond, Sigma_cond = _orfit_condition_on(
                m, U, Sigma, self.model_params.emission_mean_function, u, y, self.sv_threshold
            )
        elif self.method == 'full_svd_lofi':
            m_cond, U_cond, Sigma_cond = _lofi_full_svd_condition_on(
                m, U, Sigma, self.eta, self.model_params.emission_mean_function, 
                self.model_params.emission_cov_function, u, y, self.sv_threshold, 
                self.adaptive_variance
            )
            nu, rho = _lofi_update_noise(
                m_cond, U_cond, Sigma_cond, self.eta, self.model_params.emission_mean_function,
                u, y, nu, rho, self.adaptive_variance
            )
        elif self.method == 'orth_svd_lofi':
            m_cond, U_cond, Sigma_cond = _lofi_orth_svd_condition_on(
                m, U, Sigma, self.eta, self.model_params.emission_mean_function, 
                self.model_params.emission_cov_function, u, y, self.sv_threshold, 
                self.adaptive_variance
            )
            nu, rho = _lofi_update_noise(
                m_cond, U_cond, Sigma_cond, self.eta, self.model_params.emission_mean_function,
                u, y, nu, rho, self.adaptive_variance
            )
        return LoFiBel(mean=m_cond, basis=U_cond, sigma=Sigma_cond, nu=nu, rho=rho)


def _invert_2x2_block_matrix(M, lr_block_dim):
    """Invert a 2x2 block matrix. The matrix is assumed to be of the form:
    [[A, b],
    [b.T, c]]
    where A is a diagonal matrix.

    Args:
        M (2, 2): 2x2 block matrix.
        lr_block_dim (int): Dimension of the lower right block.
        
    Returns:
        (2, 2): Inverse of the 2x2 block matrix.
    """
    m, n = M.shape
    A = M[:m-lr_block_dim, :n-lr_block_dim]
    B = M[:m-lr_block_dim, n-lr_block_dim:]
    D = M[m-lr_block_dim:, n-lr_block_dim:]
    a = 1/jnp.diag(A)
    K_inv = jnp.linalg.inv(D - (a*B.T) @ B)

    B_inv = - (a * B.T).T @ K_inv
    A_inv = jnp.diag(a) + (a * B.T).T @ K_inv @ (a * B.T)
    C_inv = -K_inv @ (a * B.T)
    D_inv = K_inv

    return jnp.block([[A_inv, B_inv], [C_inv, D_inv]])


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


def _lofi_orth_svd_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False):
    """Condition step of the low-rank filter algorithm based on orthogonal SVD method.

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
        adaptive_variance (bool): Whether to use adaptive variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Sigma_cond (D_mem,): Posterior singular values.
    """
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    if adaptive_variance:
        A = jnp.eye(yhat.shape[0])
    else:
        R = jnp.atleast_2d(Cov_Y(m))
        L = jnp.linalg.cholesky(R)
        A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])
    S = eta*jnp.eye(W_tilde.shape[1]) + W_tilde.T @ W_tilde
    K = (H.T @ A) @ A.T - W_tilde @ (_invert_2x2_block_matrix(S, yhat.shape[0]) @ (W_tilde.T @ ((H.T @ A) @ A.T)))

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


def _lofi_full_svd_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False):
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
        adaptive_variance (bool): Whether to use adaptive variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Sigma_cond (D_mem,): Posterior singular values.
        yhat (D_obs,): Emission mean.
    """
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    if adaptive_variance:
        A = jnp.eye(yhat.shape[0])
    else:
        R = jnp.atleast_2d(Cov_Y(m))
        L = jnp.linalg.cholesky(R)
        A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])

    # Update the U matrix
    u, lamb, _ = jnp.linalg.svd(W_tilde, full_matrices=False)

    D = (lamb**2)/(eta**2 * jnp.ones(lamb.shape) + eta * lamb**2)
    K = (H.T @ A) @ A.T/eta - (D * u) @ (u.T @ ((H.T @ A) @ A.T))

    U_cond = u[:, :U.shape[1]]
    Sigma_cond = lamb[:U.shape[1]]

    m_cond = m + K @ (y - yhat)

    return m_cond, U_cond, Sigma_cond


def _lofi_update_noise(m, U, Sigma, eta, y_cond_mean, x, y, nu, rho, adaptive_variance=False):
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
    """
    if not adaptive_variance:
        return 1.0, 1.0

    m_Y = lambda w: y_cond_mean(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    H = _jacrev_2d(m_Y, m)

    lamb = (Sigma**2)/(eta**2 * jnp.ones(Sigma.shape) + eta * Sigma**2)
    HU = H @ U
    V_epi = H @ H.T/eta - (lamb * HU) @ (HU).T

    nu += yhat.shape[0]
    rho += (y - yhat).T @ jnp.linalg.pinv(jnp.eye(yhat.shape[0]) + V_epi) @ (y - yhat)

    return nu, rho


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
    memory_limit, sv_threshold, _ = inf_params
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


def low_rank_filter_orthogonal_svd(
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
    m, sv_threshold, adaptive_variance = inf_params
    U, Sigma = jnp.zeros((len(initial_mean), m)), jnp.zeros((m,))

    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    gamma = model_params.dynamics_weights
    assert isinstance(gamma, float), "Dynamics decay term must be a scalar."
    nu, rho = 0.0, 0.0
    
    # Steady-state constraint
    q = (1 - gamma**2) / eta

    def _step(carry, t):
        mean, U, Sigma, nu, rho = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Predict the next state
        pred_mean, pred_Sigma = _lofi_predict(mean, Sigma, gamma, q)

        # Condition on the emission
        filtered_mean, filtered_U, filtered_Sigma = _lofi_orth_svd_condition_on(pred_mean, U, pred_Sigma, eta, m_Y, Cov_Y, x, y, sv_threshold, adaptive_variance)

        # Update noise
        nu, rho = _lofi_update_noise(filtered_mean, filtered_U, filtered_Sigma, eta, m_Y, x, y, nu, rho, adaptive_variance)
        tau = jnp.where(jnp.isfinite(jnp.divide(rho, nu)), jnp.divide(rho, nu), 1.0)

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


def low_rank_filter_full_svd(
    model_params: RebayesParams,
    inf_params: LoFiParams,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Float[Array, "ntime input_dim"]
) -> PosteriorLoFiFiltered:
    """Low-rank filter algorithm with full SVD.

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
    m, sv_threshold, adaptive_variance = inf_params
    U, Sigma = jnp.zeros((len(initial_mean), m)), jnp.zeros((m,))

    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    gamma = model_params.dynamics_weights
    assert isinstance(gamma, float), "Dynamics decay term must be a scalar."
    nu, rho = 0.0, 0.0
    
    # Steady-state constraint
    q = (1 - gamma**2) / eta

    def _step(carry, t):
        mean, U, Sigma, nu, rho = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Predict the next state
        pred_mean, pred_Sigma = _lofi_predict(mean, Sigma, gamma, q)

        # Condition on the emission
        filtered_mean, filtered_U, filtered_Sigma = \
            _lofi_full_svd_condition_on(
                pred_mean, U, pred_Sigma, eta, m_Y, Cov_Y, x, y, sv_threshold, adaptive_variance
            )

        # Marginalize
        nu, rho = _lofi_update_noise(filtered_mean, filtered_U, filtered_Sigma, eta, m_Y, x, y, nu, rho, adaptive_variance)
        tau = jnp.where(jnp.isfinite(jnp.divide(rho, nu)), jnp.divide(rho, nu), 1.0)

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
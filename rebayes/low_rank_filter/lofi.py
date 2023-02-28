from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import jit, jacrev, vmap
from jax.lax import scan
from jaxtyping import Float, Array
import chex

from rebayes.base import RebayesParams, Rebayes, Gaussian, CovMat


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
    nobs: int=None
    obs_noise_var: float=None

    eta: CovMat=None
    gamma: float=None
    q: float=None

    @property
    def cov(self):
        """
        For large-dimensional systems,
        use at your own risk.
        """
        num_features = len(self.mean)
        D = self.sigma ** 2 / (self.eta * (self.eta + self.sigma ** 2))
        D = jnp.diag(D)

        I = jnp.eye(num_features)
        cov = I / self.eta - self.basis @ D @ self.basis.T
        return cov


class LoFiParams(NamedTuple):
    """Lightweight container for ORFit parameters.
    """
    memory_size: int
    sv_threshold: float = 0.0
    steady_state: bool = False


class PosteriorLoFiFiltered(NamedTuple):
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]
    filtered_sigmas: Float[Array, "ntime memory_size"] = None


class RebayesLoFi(Rebayes):
    def __init__(
        self,
        model_params: RebayesParams,
        lofi_params: LoFiParams,
        method: str,
    ):
        if method == 'orfit':
            self.eta = None
            self.gamma = None
            self.q = None
        elif method == 'full_svd_lofi' or method == 'orth_svd_lofi' or method == 'generalized_lofi':
            initial_cov = model_params.initial_covariance
            # assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
            self.eta = 1/initial_cov
            if method == 'generalized_lofi':
                self.eta = jnp.ones((len(model_params.initial_mean), 1)) * self.eta
            self.gamma = model_params.dynamics_weights
            # assert isinstance(self.gamma, float), "Dynamics decay term must be a scalar."
            self.steady_state = lofi_params.steady_state
            if self.steady_state and method != 'generalized_lofi':
                self.q = (1 - self.gamma**2) / self.eta
            else:
                self.q = model_params.dynamics_covariance
        else:
            raise ValueError(f"Unknown method {method}.")
        self.method = method
        self.nobs, self.obs_noise_var = 0, 0.0
        self.model_params = model_params
        self.adaptive_variance = model_params.adaptive_emission_cov
        self.m, self.sv_threshold = lofi_params
        self.U0 = jnp.zeros((len(model_params.initial_mean), self.m))
        self.Sigma0 = jnp.zeros((self.m,))
        self.alpha = model_params.dynamics_covariance_inflation_factor

    def init_bel(self):
        return LoFiBel(
            mean=self.model_params.initial_mean, basis=self.U0, sigma=self.Sigma0,
            nobs=self.nobs, obs_noise_var=self.obs_noise_var,
            eta=self.eta, gamma=self.gamma, q=self.q,
        )
    
    @partial(jit, static_argnums=(0,))
    def predict_state(self, bel):
        m, U, Sigma, nobs, obs_noise_var, eta = \
            bel.mean, bel.basis, bel.sigma, bel.nobs, bel.obs_noise_var, bel.eta
        if self.method == 'orfit':
            return bel
        elif self.method == 'orth_svd_lofi' or self.method == 'full_svd_lofi':
            m_pred, Sigma_pred, eta_pred = _lofi_predict(m, Sigma, self.gamma, self.q, eta, self.alpha, self.steady_state)
            U_pred = U
        else:
            m_pred, U_pred, Sigma_pred, eta_pred = _generalized_lofi_predict(m, U, Sigma, self.gamma, self.q, eta, self.alpha)

        return bel.replace(
            mean=m_pred, basis=U_pred, sigma=Sigma_pred,
            nobs=nobs, obs_noise_var=obs_noise_var, eta=eta_pred
        )

    @partial(jit, static_argnums=(0,))
    def predict_obs(self, bel, u):
        m, U, sigma, obs_noise_var = \
            bel.mean, bel.basis, bel.sigma, bel.obs_noise_var
        m_Y = lambda z: self.model_params.emission_mean_function(z, u)
        y_pred = jnp.atleast_1d(m_Y(m))
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def predict_obs_cov(self, bel, u):
        m, U, sigma, obs_noise_var, eta = \
            bel.mean, bel.basis, bel.sigma, bel.obs_noise_var, bel.eta
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
                R = jnp.eye(y_pred.shape[0]) * obs_noise_var
            else:
                R = jnp.atleast_2d(Cov_Y(m))
            if self.method == 'orth_svd_lofi' or self.method == 'full_svd_lofi':
                D = (sigma**2)/(eta * (eta + sigma**2))
                HU = H @ U
                V_epi = H @ H.T/eta - (D * HU) @ (HU).T
            else:
                W = U * sigma
                D = jnp.linalg.pinv(jnp.eye(W.shape[1]) +  W.T @ (W/eta))
                HW = H/eta @ W
                V_epi = H @ H.T/eta - (HW @ D) @ (HW).T
    
            Sigma_obs = V_epi + R
                    
        return Sigma_obs

    @partial(jit, static_argnums=(0,))
    def update_state(self, bel, u, y):
        m, U, Sigma, nobs, obs_noise_var, eta_cond = \
            bel.mean, bel.basis, bel.sigma, bel.nobs, bel.obs_noise_var, bel.eta
        if self.method == 'orfit':
            m_cond, U_cond, Sigma_cond = _orfit_condition_on(
                m, U, Sigma, self.model_params.emission_mean_function, u, y, self.sv_threshold
            )
        else:
            nobs, obs_noise_var = _lofi_estimate_noise(
                m, self.model_params.emission_mean_function,
                u, y, nobs, obs_noise_var, self.adaptive_variance
            )
            if self.method == 'full_svd_lofi':
                m_cond, U_cond, Sigma_cond = _lofi_full_svd_condition_on(
                    m, U, Sigma, eta_cond, self.model_params.emission_mean_function, 
                    self.model_params.emission_cov_function, u, y, self.sv_threshold, 
                    self.adaptive_variance, obs_noise_var
                )
            elif self.method == 'orth_svd_lofi':
                m_cond, U_cond, Sigma_cond = _lofi_orth_svd_condition_on(
                    m, U, Sigma, eta_cond, self.model_params.emission_mean_function, 
                    self.model_params.emission_cov_function, u, y, self.sv_threshold, 
                    self.adaptive_variance, obs_noise_var, nobs
                )
            elif self.method == 'generalized_lofi':
                m_cond, U_cond, Sigma_cond, eta_cond = _generalized_lofi_condition_on(
                    m, U, Sigma, eta_cond, self.model_params.emission_mean_function, 
                    self.model_params.emission_cov_function, u, y, self.sv_threshold, 
                    self.adaptive_variance, obs_noise_var
                )

        return bel.replace(
            mean=m_cond, basis=U_cond, sigma=Sigma_cond, nobs=nobs, 
            obs_noise_var=obs_noise_var, eta=eta_cond
        )


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


def _lofi_orth_svd_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False, obs_noise_var=1.0, key=0):
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
        key (int): Random key.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Sigma_cond (D_mem,): Posterior singular values.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    if adaptive_variance:
        R = jnp.eye(yhat.shape[0]) * obs_noise_var
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

    perm = jr.permutation(key, yhat.shape[0])
    (U_cond, Sigma_cond), _ = scan(_update_basis, (U, Sigma), perm)

    return m_cond, U_cond, Sigma_cond


def _lofi_full_svd_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False, obs_noise_var=1.0):
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
    """
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    if adaptive_variance:
        R = jnp.eye(yhat.shape[0]) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    L = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])

    # Update the U matrix
    u, lamb, _ = jnp.linalg.svd(W_tilde, full_matrices=False)

    D = (lamb**2)/(eta**2 + eta * lamb**2)
    K = (H.T @ A) @ A.T/eta - (D * u) @ (u.T @ ((H.T @ A) @ A.T))

    U_cond = u[:, :U.shape[1]]
    Sigma_cond = lamb[:U.shape[1]]

    m_cond = m + K @ (y - yhat)

    return m_cond, U_cond, Sigma_cond


def _generalized_lofi_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False, obs_noise_var=1.0):
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
    """
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    if adaptive_variance:
        R = jnp.eye(yhat.shape[0]) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    L = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])
    
    # Update the U matrix
    u, lamb, _ = jnp.linalg.svd(W_tilde, full_matrices=False)
    U_cond, U_extra = u[:, :U.shape[1]], u[:, U.shape[1]:]
    Sigma_cond, Sigma_extra = lamb[:U.shape[1]], lamb[U.shape[1]:]
    W_extra = Sigma_extra * U_extra
    eta_cond = eta + jnp.einsum('ij,ij->i', W_extra, W_extra)[:, jnp.newaxis]
    
    D = jnp.linalg.pinv(jnp.eye(W_tilde.shape[1]) + W_tilde.T @ (W_tilde/eta))
    K = (H.T @ A) @ A.T/eta - (W_tilde/eta @ D) @ ((W_tilde/eta).T @ (H.T @ A) @ A.T)
    m_cond = m + K @ (y - yhat)
    
    return m_cond, U_cond, Sigma_cond, eta_cond


def _lofi_estimate_noise(m, y_cond_mean, u, y, nobs, obs_noise_var, adaptive_variance=False):
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


def _lofi_predict(m, Sigma, gamma, q, eta, alpha=0.0, steady_state=False):
    """Predict step of the low-rank filter algorithm.

    Args:
        m (D_hid,): Prior mean.
        Sigma (D_mem,): Prior singluar values.
        gamma (float): Dynamics decay factor.
        q (float): Dynamics noise factor.
        eta (float): Prior precision.
        alpha (float): Covariance inflation factor.
        steady_state (bool): Whether to use steady-state dynamics.

    Returns:
        m_pred (D_hid,): Predicted mean.
        Sigma_pred (D_mem,): Predicted singular values.
        eta_pred (float): Predicted precision.
    """
    m_pred = gamma * m
    Sigma_pred = jnp.sqrt((gamma**2 * Sigma**2)/((gamma**2 + q * eta) * (gamma**2 + q*eta + q*Sigma**2)))
    
    if steady_state:
        eta_pred = eta
    else:
        eta_pred = eta/(gamma**2 + q*eta)
        # eta_pred = eta_pred / (1 + alpha)
    
    # Covariance inflation
    Sigma_pred = Sigma_pred / jnp.sqrt(1 + alpha)

    return m_pred, Sigma_pred, eta_pred


def _generalized_lofi_predict(m, U, Sigma, gamma, q, eta, alpha=0.0):
    """Predict step of the generalized low-rank filter algorithm.

    Args:
        m (D_hid,): Prior mean.
        Sigma (D_mem,): Prior singluar values.
        gamma (float): Dynamics decay factor.
        q (float): Dynamics noise factor.
        eta (D_hid,): Prior precision.
        alpha (float): Covariance inflation factor.

    Returns:
        m_pred (D_hid,): Predicted mean.
        U_pred (D_hid, D_mem,): Predicted basis.
        Sigma_pred (D_mem,): Predicted singular values.
        eta_pred (float): Predicted precision.
    """
    m_pred = gamma * m
    
    # eta_pred = 1/(gamma**2 * (1+alpha)/eta + q)
    eta_pred = 1/(gamma**2/eta + q)
        
    W = U * Sigma
    chol_factor = jnp.linalg.cholesky(
        jnp.linalg.pinv(
            jnp.eye(W.shape[1]) + W.T @ (1/(gamma**2 * (1+alpha)/q + eta) * W) 
        )
    )
    W_pred = gamma * jnp.sqrt(1+alpha)/q * (1/(gamma**2 * (1+alpha)/q + eta) * W) @ chol_factor
    U_pred, Sigma_pred, _ = jnp.linalg.svd(W_pred, full_matrices=False)
    
    return m_pred, U_pred, Sigma_pred, eta_pred
    

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
    q, alpha = model_params.dynamics_covariance, model_params.dynamics_covariance_inflation_factor
    adaptive_variance = model_params.adaptive_emission_cov
    assert isinstance(initial_cov, float) and initial_cov > 0, "Initial covariance must be a positive scalar."
    eta = 1/initial_cov
    m, sv_threshold = inf_params
    U, Sigma = jnp.zeros((len(initial_mean), m)), jnp.zeros((m,))

    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    gamma = model_params.dynamics_weights
    assert isinstance(gamma, float), "Dynamics decay term must be a scalar."
    nobs, obs_noise_var = 0, 0.0

    def _step(carry, t):
        mean, U, Sigma, eta, nobs, obs_noise_var = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Predict the next state
        pred_mean, pred_Sigma, pred_eta = _lofi_predict(mean, Sigma, gamma, q, eta, alpha)

        # Condition on the emission
        filtered_mean, filtered_U, filtered_Sigma = _lofi_orth_svd_condition_on(pred_mean, U, pred_Sigma, pred_eta, m_Y, Cov_Y, x, y, sv_threshold, adaptive_variance, obs_noise_var)

        # Update noise
        nobs, obs_noise_var = _lofi_estimate_noise(filtered_mean, filtered_U, m_Y, x, y, nobs, obs_noise_var, adaptive_variance)

        return (pred_mean, filtered_U, pred_Sigma, nobs, obs_noise_var), (filtered_mean, filtered_U, filtered_Sigma)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma, eta, nobs, obs_noise_var)
    _, (filtered_means, filtered_bases, filtered_Sigmas) = scan(_step, carry, jnp.arange(len(inputs)))
    filtered_posterior = PosteriorLoFiFiltered(
        filtered_means=filtered_means, 
        filtered_bases=filtered_bases,
        filtered_sigmas=filtered_Sigmas,
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
    q, alpha = model_params.dynamics_covariance, model_params.dynamics_covariance_inflation_factor
    adaptive_variance = model_params.adaptive_emission_cov
    m, sv_threshold = inf_params
    U, Sigma = jnp.zeros((len(initial_mean), m)), jnp.zeros((m,))

    m_Y, Cov_Y = model_params.emission_mean_function, model_params.emission_cov_function
    gamma = model_params.dynamics_weights
    assert isinstance(gamma, float), "Dynamics decay term must be a scalar."
    nobs, obs_noise_var = 0, 0.0

    def _step(carry, t):
        mean, U, Sigma, eta, nobs, obs_noise_var = carry

        # Get input and emission and compute Jacobians
        x, y = inputs[t], emissions[t]

        # Predict the next state
        pred_mean, pred_Sigma, pred_eta = _lofi_predict(mean, Sigma, gamma, q, eta, alpha)

        # Condition on the emission
        filtered_mean, filtered_U, filtered_Sigma = \
            _lofi_full_svd_condition_on(
                pred_mean, U, pred_Sigma, pred_eta, m_Y, Cov_Y, x, y, sv_threshold, adaptive_variance, obs_noise_var
            )

        # Update noise
        nobs, obs_noise_var = _lofi_estimate_noise(filtered_mean, filtered_U, m_Y, x, y, nobs, obs_noise_var, adaptive_variance)

        return (pred_mean, filtered_U, pred_Sigma, nobs, obs_noise_var), (filtered_mean, filtered_U, filtered_Sigma)
    
    # Run ORFit
    carry = (initial_mean, U, Sigma, eta, nobs, obs_noise_var)
    _, (filtered_means, filtered_bases, filtered_Sigmas) = scan(_step, carry, jnp.arange(len(inputs)))
    filtered_posterior = PosteriorLoFiFiltered(
        filtered_means=filtered_means, 
        filtered_bases=filtered_bases,
        filtered_sigmas=filtered_Sigmas,
    )

    return filtered_posterior
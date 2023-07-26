from typing import Tuple, Callable

from jax import jacrev, jit
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array


# Helper functions -------------------------------------------------------------

_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))
_normalize = lambda v: jnp.where(v.any(), v / jnp.linalg.norm(v), jnp.zeros(shape=v.shape))
_vec_pinv = lambda v: jnp.where(v != 0, 1/jnp.array(v), 0) # Vector pseudo-inverse


def _invert_2x2_block_matrix(
    M: Float[Array, "m n"],
    lr_block_dim: int
) -> Float[Array, "m n"]:
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


def _fast_svd(
    M: Float[Array, "m n"],
) -> Tuple[Float[Array, "m k"], Float[Array, "k"]]:
    """Singular value decomposition.

    Args:
        M (m, n): Matrix to decompose.

    Returns:
        U (m, k): Left singular vectors.
        S (k,): Singular values.
    """
    U, S, _ = jnp.linalg.svd(M.T @ M, full_matrices = False, hermitian = True)
    U = M @ (U * _vec_pinv(jnp.sqrt(S)))
    S = jnp.sqrt(S)
    
    return U, S
    


# Common inference functions ---------------------------------------------------

def _lofi_estimate_noise(
    m: Float[Array, "state_dim"],
    y_cond_mean: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    nobs: int,
    obs_noise_var: float,
    adaptive_variance: bool = False
) -> Tuple[int, float]:
    """Estimate observation noise based on empirical residual errors.

    Args:
        m (D_hid,): Prior mean.
        y_cond_mean (Callable): Conditional emission mean function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        nobs (int): Number of observations seen so far.
        obs_noise_var (float): Current estimate of observation noise.
        adaptive_variance (bool): Whether to use adaptive variance.

    Returns:
        nobs_est (int): Updated number of observations seen so far.
        obs_noise_var_est (float): Updated estimate of observation noise.
    """
    nobs_est = nobs + 1
    if not adaptive_variance:
        return nobs_est, 0.0

    m_Y = lambda w: y_cond_mean(w, x)
    yhat = jnp.atleast_1d(m_Y(m))
    
    sqerr = ((yhat - y).T @ (yhat - y)).squeeze() / yhat.shape[0]
    obs_noise_var_est = jnp.max(jnp.array([1e-6, obs_noise_var + 1/nobs_est * (sqerr - obs_noise_var)]))

    return nobs_est, obs_noise_var_est


# Spherical LOFI ---------------------------------------------------------------

def _lofi_spherical_cov_inflate(
    m0: Float[Array, "state_dim"],
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    eta: float,
    alpha: float,
    inflation: str = "bayesian"
):
    """Inflate the spherical posterior covariance matrix.

    Args:
        m0 (D_hid,): Prior predictive mean.
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior signular values.
        eta (float): Prior precision.
        alpha (float): Covariance inflation factor.
        inflation (str, optional): Type of inflation. Defaults to 'bayesian'.

    Returns:
        m_infl (D_hid,): Post-inflation mean.
        U_infl (D_hid, D_mem,): Post-inflation basis.
        Lambda_infl (D_mem,): Post-inflation singular values.
        eta_infl (float): Post-inflation precision.
    """    
    Lambda_infl = Lambda / jnp.sqrt(1+alpha)
    U_infl = U
    W_infl = U_infl * Lambda_infl
    
    if inflation == 'bayesian':
        eta_infl = eta
        G = jnp.linalg.pinv(jnp.eye(W_infl.shape[1]) +  (W_infl.T @ (W_infl/eta_infl)))
        e = (m0 - m)
        K = e - ((W_infl/eta_infl) @ G) @ (W_infl.T @ e)
        m_infl = m + alpha/(1+alpha) * K.ravel()
    elif inflation == 'simple':
        eta_infl = eta/(1+alpha)
        m_infl = m
    elif inflation == 'hybrid':
        eta_infl = eta
        m_infl = m
    
    return m_infl, U_infl, Lambda_infl, eta_infl


def _lofi_spherical_cov_predict(
    m0: Float[Array, "state_dim"],
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    gamma: float,
    q: float,
    eta: float,
    steady_state: bool = False
):
    """Predict step of the low-rank filter with spherical covariance matrix.

    Args:
        m0 (D_hid,): Prior predictive mean.
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singluar values.
        gamma (float): Dynamics decay factor.
        q (float): Dynamics noise factor.
        eta (float): Prior precision.
        alpha (float): Covariance inflation factor.
        steady_state (bool): Whether to use steady-state dynamics.

    Returns:
        m0_pred (D_hid,): Predicted predictive mean.
        m_pred (D_hid,): Predicted mean.
        U_pred (D_hid, D_mem,): Predicted basis.
        Lambda_pred (D_mem,): Predicted singular values.
        eta_pred (float): Predicted precision.
    """
    # Mean prediction
    m0_pred = gamma*m0
    m_pred = gamma*m

    # Covariance prediction
    U_pred = U
    
    if steady_state:
        eta_pred = eta
        Lambda_pred = jnp.sqrt(
            (gamma**2 * Lambda**2) /
            (1 + q*Lambda**2)
        )
    else:
        eta_pred = eta/(gamma**2 + q*eta)
        Lambda_pred = jnp.sqrt(
            (gamma**2 * Lambda**2) /
            ((gamma**2 + q*eta) * (gamma**2 + q*eta + q*Lambda**2))
        )

    return m0_pred, m_pred, U_pred, Lambda_pred, eta_pred


def _lofi_spherical_cov_svd_free_predict(
    m0: Float[Array, "state_dim"],
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    gamma: float,
    q: float,
    eta: float,
    steady_state: bool = False
):
    """Predict step of the low-rank filter with spherical covariance matrix.

    Args:
        m0 (D_hid,): Prior predictive mean.
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singluar values.
        gamma (float): Dynamics decay factor.
        q (float): Dynamics noise factor.
        eta (float): Prior precision.
        alpha (float): Covariance inflation factor.
        steady_state (bool): Whether to use steady-state dynamics.

    Returns:
        m0_pred (D_hid,): Predicted predictive mean.
        m_pred (D_hid,): Predicted mean.
        U_pred (D_hid, D_mem,): Predicted basis.
        Lambda_pred (D_mem,): Predicted singular values.
        eta_pred (float): Predicted precision.
    """
    # Mean prediction
    m0_pred = gamma*m0
    m_pred = gamma*m

    # Covariance prediction
    denom = gamma**2 + q*eta
    eta_pred = eta/denom
    C = jnp.linalg.pinv(
        jnp.diag(1/Lambda**2) + (q/denom) * U.T @ U
    )
    W = (gamma/denom) * U @ jnp.linalg.cholesky(C)
    Lambda_pred = jnp.sqrt(jnp.einsum("ji,ji->i", W, W))
    U_pred = W / Lambda_pred

    return m0_pred, m_pred, U_pred, Lambda_pred, eta_pred


def _lofi_spherical_cov_condition_on(
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    eta: float,
    y_cond_mean: Callable,
    y_cond_cov: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    adaptive_variance: bool = False,
    obs_noise_var: float = 1.0,
):
    """Condition step of the low-rank filter with spherical covariance matrix.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        eta (float): Prior precision. 
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        obs_noise_var (float): Observation noise variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
    """
    P, L = U.shape
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    C = yhat.shape[0]
    
    if adaptive_variance:
        R = jnp.eye(C) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(C))[0].T
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Lambda * U, (H.T @ A).reshape(P, -1)])

    # Update the U matrix
    u, lamb = _fast_svd(W_tilde)

    D = (lamb**2)/(eta**2 + eta * lamb**2)
    K = (H.T @ A) @ A.T/eta - (D * u) @ (u.T @ ((H.T @ A) @ A.T))

    U_cond = u[:, :L]
    Lambda_cond = lamb[:L]

    m_cond = m + K @ (y - yhat)

    return m_cond, U_cond, Lambda_cond


def _lofi_spherical_cov_svd_free_condition_on(
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    eta: float,
    y_cond_mean: Callable,
    y_cond_cov: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    adaptive_variance: bool = False,
    obs_noise_var: float = 1.0,
):
    """Condition step of the low-rank filter with spherical covariance matrix.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        eta (float): Prior precision. 
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        obs_noise_var (float): Observation noise variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
    """
    P, L = U.shape
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    yhat = jnp.atleast_1d(m_Y(m))
    C = yhat.shape[0]
    
    if adaptive_variance:
        R = jnp.eye(C) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(C))[0].T
    H = _jacrev_2d(m_Y, m)
    AH = A.T @ H
    
    Lambda_plus = jnp.sqrt(jnp.einsum("ij,ij->i", AH, AH))
    Lambda_tilde = jnp.hstack([Lambda, Lambda_plus])
    U_tilde = jnp.hstack([U, AH.T/Lambda_plus])

    G = jnp.linalg.pinv(jnp.diag(eta/(Lambda_tilde**2)) + U_tilde.T @ U_tilde)
    K = (H.T @ A) @ A.T/eta - ((U_tilde/eta) @ G) @ ((U_tilde/eta).T @ (H.T @ A) @ A.T)
    m_cond = m + K @ (y - yhat)
    
    sorted_order = jnp.argsort(-jnp.abs(Lambda_tilde))
    U_cond = U_tilde[:, sorted_order[:L]]
    Lambda_cond = Lambda_tilde[sorted_order[:L]]

    return m_cond, U_cond, Lambda_cond


# Diagonal LOFI ----------------------------------------------------------------

def _lofi_diagonal_cov_inflate(
    m0: Float[Array, "state_dim"],
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    eta: float,
    Ups: Float[Array, "state_dim"],
    alpha: float,
    inflation: str = "bayesian"
):
    """Inflate the diagonal posterior covariance matrix.

    Args:
        m0 (D_hid,): Prior predictive mean.
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior signular values.
        eta (float): Prior precision.
        Ups (D_hid,): Prior diagonal covariance.
        alpha (float): Covariance inflation factor.
        inflation (str, optional): Type of inflation. Defaults to 'bayesian'.

    Returns:
        m_infl (D_hid,): Post-inflation mean.
        U_infl (D_hid, D_mem,): Post-inflation basis.
        Lambda_infl (D_mem,): Post-inflation singular values.
        Ups_infl (D_hid,): Post-inflation diagonal covariance.
    """
    P, L = U.shape
    W = U * Lambda
    
    if inflation == 'bayesian':
        W_infl = W/jnp.sqrt(1+alpha)
        Ups_infl = Ups/(1+alpha) + alpha*eta/(1+alpha)
        G = jnp.linalg.pinv(jnp.eye(L) +  (W_infl.T @ (W_infl/Ups_infl)))
        e = (m0 - m)
        K = 1/Ups_infl.ravel() * (e - (W_infl @ G) @ ((W_infl/Ups_infl).T @ e))
        m_infl = m + alpha*eta/(1+alpha) * K
    elif inflation == 'simple':
        W_infl = W/jnp.sqrt(1+alpha)
        Ups_infl = Ups/(1+alpha)
        m_infl = m
    elif inflation == 'hybrid':
        W_infl = W/jnp.sqrt(1+alpha)
        Ups_infl = Ups/(1+alpha) + alpha*eta/(1+alpha)
        m_infl = m
    U_infl, Lambda_infl = W_infl, jnp.ones(L)
    
    return  m_infl, U_infl, Lambda_infl, Ups_infl


def _lofi_diagonal_cov_predict(
    m0: Float[Array, "state_dim"],
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    gamma: float,
    q: float,
    eta: float,
    Ups: Float[Array, "state_dim"],
):
    """Predict step of the low-rank filter with diagonal covariance matrix.

    Args:
        m0 (D_hid,): Initial mean.
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singluar values.
        gamma (float): Dynamics decay factor.
        q (float): Dynamics noise factor.
        eta (float): Prior precision.
        Ups (D_hid,): Prior diagonal covariance.

    Returns:
        m0_pred (D_hid,): Predicted predictive mean.
        m_pred (D_hid,): Predicted mean.
        U_pred (D_hid, D_mem,): Predicted basis.
        Lambda_pred (D_mem,): Predicted singular values.
        eta_pred (float): Predicted precision.
        Ups_pred (D_hid,): Predicted diagonal covariance.
    """
    P, L = U.shape
    
    # Mean prediction
    m0_pred = gamma*m0
    W = U * Lambda
    m_pred = gamma*m

    # Covariance prediction
    eta_pred = eta/(gamma**2 + q*eta)
    Ups_pred = 1/(gamma**2/Ups + q)
    C = jnp.linalg.pinv(jnp.eye(L) + q*W.T @ (W*(Ups_pred/Ups)))
    W_pred = gamma*(Ups_pred/Ups)*W @ jnp.linalg.cholesky(C)
    U_pred, Lambda_pred = W_pred, jnp.ones(L)
    
    return m0_pred, m_pred, U_pred, Lambda_pred, eta_pred, Ups_pred


def _lofi_diagonal_cov_condition_on(
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    Ups: Float[Array, "state_dim"],
    y_cond_mean: Callable,
    y_cond_cov: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    adaptive_variance: bool = False,
    obs_noise_var: float = 1.0,
):
    """Condition step of the low-rank filter with diagonal covariance matrix.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        Ups (D_hid): Prior precision. 
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        obs_noise_var (float): Observation noise variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
        Ups_cond (D_hid,): Posterior precision.
    """
    P, L = U.shape
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    H = _jacrev_2d(m_Y, m)
    yhat = jnp.atleast_1d(m_Y(m))
    C = yhat.shape[0]
    
    if adaptive_variance:
        R = jnp.eye(C) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(C))[0].T
    W_tilde = jnp.hstack([Lambda * U, (H.T @ A).reshape(P, -1)])
    
    # Update the U matrix
    u, lamb = _fast_svd(W_tilde)
    
    U_cond, U_extra = u[:, :L], u[:, L:]
    Lambda_cond, Lambda_extra = lamb[:L], lamb[L:]
    W_extra = Lambda_extra * U_extra
    Ups_cond = Ups + jnp.einsum('ij,ij->i', W_extra, W_extra)[:, jnp.newaxis]
    
    G = jnp.linalg.pinv(jnp.eye(W_tilde.shape[1]) + W_tilde.T @ (W_tilde/Ups))
    K = (H.T @ A) @ A.T/Ups - (W_tilde/Ups @ G) @ ((W_tilde/Ups).T @ (H.T @ A) @ A.T)
    m_cond = m + K @ (y - yhat)
    
    return m_cond, U_cond, Lambda_cond, Ups_cond


def _lofi_diagonal_cov_svd_free_condition_on(
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    Ups: Float[Array, "state_dim"],
    y_cond_mean: Callable,
    y_cond_cov: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    adaptive_variance: bool = False,
    obs_noise_var: float = 1.0,
):
    """Condition step of the SVD-free low-rank filter with diagonal covariance matrix.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        Ups (D_hid): Prior precision. 
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        obs_noise_var (float): Observation noise variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
        Ups_cond (D_hid,): Posterior precision.
    """
    P, L = U.shape
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)
    
    H = _jacrev_2d(m_Y, m)
    yhat = jnp.atleast_1d(m_Y(m))
    C = yhat.shape[0]
    
    if adaptive_variance:
        R = jnp.eye(C) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(C))[0].T
    AH = A.T @ H
    
    Lambda_plus = jnp.sqrt(jnp.einsum("ij,ij->i", AH/Ups.T, AH))
    Lambda_tilde = jnp.hstack([Lambda, Lambda_plus])
    U_tilde = jnp.hstack([U, AH.T/Lambda_plus])
    
    G = jnp.linalg.pinv(jnp.diag(1/(Lambda_tilde**2)) + U_tilde.T @ (U_tilde/Ups))
    K = (H.T @ A) @ A.T/Ups - ((U_tilde/Ups) @ G) @ ((U_tilde/Ups).T @ (H.T @ A) @ A.T)
    m_cond = m + K @ (y - yhat)
    
    sorted_order = jnp.argsort(-jnp.abs(Lambda_tilde))
    U_cond, U_extra = U_tilde[:, sorted_order[:L]], U_tilde[:, sorted_order[L:]]
    Lambda_cond, Lambda_extra = Lambda_tilde[sorted_order[:L]], Lambda_tilde[sorted_order[L:]]
    W_extra = Lambda_extra * U_extra
    Ups_cond = Ups + jnp.einsum("ij,ij->i", W_extra, W_extra)[:, jnp.newaxis]
    
    return m_cond, U_cond, Lambda_cond, Ups_cond


def _replay_lofi_diagonal_cov_condition_on(
    m: Float[Array, "state_dim"],
    m_lin: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    Ups: Float[Array, "state_dim"],
    y_cond_mean: Callable,
    y_cond_cov: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    adaptive_variance: bool = False,
    obs_noise_var: float = 1.0,
):
    """Condition step of the low-rank filter with diagonal covariance matrix.

    Args:
        m (D_hid,): Prior mean.
        m_lin (D_hid,): Linearization point for mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        Ups (D_hid): Prior precision. 
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        obs_noise_var (float): Observation noise variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
        Ups_cond (D_hid,): Posterior precision.
    """
    P, L = U.shape
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, w, x)
    
    H = _jacrev_2d(m_Y, m_lin)
    # yhat = jnp.atleast_1d(m_Y(m_lin)) + H @ (m - m_lin)
    yhat = jnp.atleast_1d(m_Y(m))
    C = yhat.shape[0]
    
    if adaptive_variance:
        R = jnp.eye(C) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(C))[0].T
    W_tilde = jnp.hstack([Lambda * U, (H.T @ A).reshape(P, -1)])
    
    # Update the U matrix
    u, lamb = _fast_svd(W_tilde)
    
    U_cond, U_extra = u[:, :L], u[:, L:]
    Lambda_cond, Lambda_extra = lamb[:L], lamb[L:]
    W_extra = Lambda_extra * U_extra
    Ups_cond = Ups + jnp.einsum('ij,ij->i', W_extra, W_extra)[:, jnp.newaxis]
    
    G = jnp.linalg.pinv(jnp.eye(W_tilde.shape[1]) + W_tilde.T @ (W_tilde/Ups))
    K = (H.T @ A) @ A.T/Ups - (W_tilde/Ups @ G) @ ((W_tilde/Ups).T @ (H.T @ A) @ A.T)
    m_cond = m + K @ (y - yhat)
    
    return m_cond, U_cond, Lambda_cond, Ups_cond


def _replay_lofi_diagonal_cov_svd_free_condition_on(
    m: Float[Array, "state_dim"],
    m_lin: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    Ups: Float[Array, "state_dim"],
    y_cond_mean: Callable,
    y_cond_cov: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    adaptive_variance: bool = False,
    obs_noise_var: float = 1.0,
):
    """Condition step of the SVD-free low-rank filter with diagonal covariance matrix.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        Ups (D_hid): Prior precision. 
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        obs_noise_var (float): Observation noise variance.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
        Ups_cond (D_hid,): Posterior precision.
    """
    P, L = U.shape
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, m_lin, x)
    
    H = _jacrev_2d(m_Y, m_lin)
    yhat = jnp.atleast_1d(m_Y(m_lin)) + H @ (m - m_lin)
    C = yhat.shape[0]
    
    if adaptive_variance:
        R = jnp.eye(C) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(C))[0].T
    AH = A.T @ H
    
    Lambda_plus = jnp.sqrt(jnp.einsum("ij,ij->i", AH/Ups.T, AH))
    Lambda_tilde = jnp.hstack([Lambda, Lambda_plus])
    U_tilde = jnp.hstack([U, AH.T/Lambda_plus])
    
    G = jnp.linalg.pinv(jnp.diag(1/(Lambda_tilde**2)) + U_tilde.T @ (U_tilde/Ups))
    K = (H.T @ A) @ A.T/Ups - ((U_tilde/Ups) @ G) @ ((U_tilde/Ups).T @ (H.T @ A) @ A.T)
    m_cond = m + K @ (y - yhat)
    
    sorted_order = jnp.argsort(-jnp.abs(Lambda_tilde))
    U_cond, U_extra = U_tilde[:, sorted_order[:L]], U_tilde[:, sorted_order[L:]]
    Lambda_cond, Lambda_extra = Lambda_tilde[sorted_order[:L]], Lambda_tilde[sorted_order[L:]]
    W_extra = Lambda_extra * U_extra
    Ups_cond = Ups + jnp.einsum("ij,ij->i", W_extra, W_extra)[:, jnp.newaxis]
    
    return m_cond, U_cond, Lambda_cond, Ups_cond


# Orthogonal LOFI --------------------------------------------------------------

def _lofi_orth_condition_on(
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    eta: float,
    y_cond_mean: Callable,
    y_cond_cov: Callable,
    x: Float[Array, "input_dim"],
    y: Float[Array, "obs_dim"],
    adaptive_variance: bool = False,
    obs_noise_var: float = 1.0,
    key: int = 0
):
    """Condition step of the low-rank filter algorithm based on orthogonal SVD method.

    Args:
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singular values.
        eta (float): Prior precision.
        y_cond_mean (Callable): Conditional emission mean function.
        y_cond_cov (Callable): Conditional emission covariance function.
        x (D_in,): Control input.
        y (D_obs,): Emission.
        adaptive_variance (bool): Whether to use adaptive variance.
        obs_noise_var (float): Observation noise variance.
        key (int): Random key.

    Returns:
        m_cond (D_hid,): Posterior mean.
        U_cond (D_hid, D_mem,): Posterior basis.
        Lambda_cond (D_mem,): Posterior singular values.
    """
    if isinstance(key, int) or len(key.shape) < 1:
        key = jr.PRNGKey(key)
    P, L = U.shape
    
    m_Y = lambda w: y_cond_mean(w, x)
    Cov_Y = lambda w: y_cond_cov(w, x)

    yhat = jnp.atleast_1d(m_Y(m))    
    C = yhat.shape[0]
    
    if adaptive_variance:
        R = jnp.eye(C) * obs_noise_var
    else:
        R = jnp.atleast_2d(Cov_Y(m))
    R_chol = jnp.linalg.cholesky(R)
    A = jnp.linalg.lstsq(R_chol, jnp.eye(C))[0].T
    H = _jacrev_2d(m_Y, m)
    W_tilde = jnp.hstack([Lambda * U, (H.T @ A).reshape(P, -1)])
    S = eta*jnp.eye(W_tilde.shape[1]) + W_tilde.T @ W_tilde
    K = (H.T @ A) @ A.T - W_tilde @ (_invert_2x2_block_matrix(S, C) @ (W_tilde.T @ ((H.T @ A) @ A.T)))

    # Update the basis and singular values
    def _update_basis(carry, i):
        U, Lambda = carry
        U_tilde = (H.T - U @ (U.T @ H.T)) @ A
        v = U_tilde[:, i]
        u = _normalize(v)
        U_cond = jnp.where(Lambda.min() < u @ v, U.at[:, Lambda.argmin()].set(u), U)
        Lambda_cond = jnp.where(Lambda.min() < u @ v, Lambda.at[Lambda.argmin()].set(u.T @ v), Lambda)
        
        return (U_cond, Lambda_cond), (U_cond, Lambda_cond)

    perm = jr.permutation(key, C)
    (U_cond, Lambda_cond), _ = scan(_update_basis, (U, Lambda), perm)
    
    # Update the mean
    m_cond = m + K/eta @ (y - yhat)

    return m_cond, U_cond, Lambda_cond


# Gradient LOFI ----------------------------------------------------------------

def _lofi_gradient_diagonal_cov_predict(
    m0: Float[Array, "state_dim"],
    m: Float[Array, "state_dim"],
    U: Float[Array, "state_dim memory_size"],
    Lambda: Float[Array, "memory_size"],
    gamma: float,
    q: float,
    eta: float,
    Ups: Float[Array, "state_dim"],
):
    """Predict step of the low-rank filter with diagonal covariance matrix.

    Args:
        m0 (D_hid,): Initial mean.
        m (D_hid,): Prior mean.
        U (D_hid, D_mem,): Prior basis.
        Lambda (D_mem,): Prior singluar values.
        gamma (float): Dynamics decay factor.
        q (float): Dynamics noise factor.
        eta (float): Prior precision.
        Ups (D_hid,): Prior diagonal covariance.

    Returns:
        m0_pred (D_hid,): Predicted predictive mean.
        m_pred (D_hid,): Predicted mean.
        U_pred (D_hid, D_mem,): Predicted basis.
        Lambda_pred (D_mem,): Predicted singular values.
        eta_pred (float): Predicted precision.
        Ups_pred (D_hid,): Predicted diagonal covariance.
    """
    P, L = U.shape
    
    # Mean prediction
    m0_pred = gamma*m0
    W = U * Lambda
    m_pred = gamma*m

    # Covariance prediction
    eta_pred = eta/(gamma**2 + q*eta)
    Ups_pred = 1/(gamma**2/Ups + q)
    C = jnp.linalg.pinv(jnp.eye(L) + q*W.T @ (W*(Ups_pred/Ups)))
    W_pred = gamma*(Ups_pred/Ups)*W @ jnp.linalg.cholesky(C)
    U_pred, Lambda_pred = W_pred, jnp.ones(L)
    S_pred = (W/Ups).T
    T_pred = gamma**2 * jnp.linalg.pinv(jnp.eye(L) + W.T @ (W/Ups)) @ S_pred
    
    return m0_pred, m_pred, U_pred, Lambda_pred, eta_pred, Ups_pred, S_pred, T_pred
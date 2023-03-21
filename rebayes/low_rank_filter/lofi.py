from functools import partial
from typing import Union, Any

import chex
from jax import jit
import jax.numpy as jnp
from jaxtyping import Float, Array

from rebayes.base import RebayesParams, Rebayes, CovMat


INFLATION_METHODS = [
    'bayesian', 
    'simple', 
    'hybrid',
]


@chex.dataclass
class LoFiBel:
    pp_mean: chex.Array
    mean: chex.Array
    basis: chex.Array
    svs: chex.Array
    eta: float
    gamma: float
    q: float
    
    Ups: CovMat = None
    nobs: int = 0
    obs_noise_var: float = 0.0

    # @property
    # def cov(self):
    #     """
    #     For large-dimensional systems,
    #     use at your own risk.
    #     """
    #     Lambda = self.singular_values
    #     num_features = len(self.mean)
    #     D = Lambda ** 2 / (self.eta * (self.eta + Lambda ** 2))
    #     D = jnp.diag(D)

    #     I = jnp.eye(num_features)
    #     cov = I / self.eta - self.basis @ D @ self.basis.T
    #     return cov


@chex.dataclass
class LoFiParams:
    """Lightweight container for LOFI parameters.
    """
    memory_size: int
    steady_state: bool = False
    inflation: str = 'bayesian'


@chex.dataclass
class PosteriorLoFiFiltered:
    """Marginals of the Gaussian filtering posterior.
    """
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_bases: Float[Array, "ntime state_dim memory_size"]
    filtered_svs: Float[Array, "ntime memory_size"] = None


class RebayesLoFi(Rebayes):
    def __init__(
        self,
        model_params: RebayesParams,
        lofi_params: LoFiParams,
    ):
        # Check inflation type
        if lofi_params.inflation not in INFLATION_METHODS:
            raise ValueError(f"Unknown inflation method: {lofi_params.inflation}.")
        
        self.model_params = model_params
        self.lofi_params = lofi_params
        
    def init_bel(self):
        pp_mean = self.model_params.initial_mean # Predictive prior mean
        init_mean = self.model_params.initial_mean # Initial mean
        memory_size = self.lofi_params.memory_size
        init_basis = jnp.zeros((len(init_mean), memory_size)) # Initial basis
        init_svs = jnp.zeros(memory_size) # Initial singular values
        init_eta = 1 / self.model_params.initial_covariance # Initial precision
        gamma = self.model_params.dynamics_weights # Dynamics weights
        q = self.model_params.dynamics_covariance # Dynamics covariance
        if self.lofi_params.steady_state: # Steady-state constraint
            q = self.steady_state_constraint(init_eta, gamma)
        init_Ups = jnp.ones((len(init_mean), 1)) * init_eta
        
        return LoFiBel(
            pp_mean = pp_mean,
            mean = init_mean,
            basis = init_basis,
            svs = init_svs,
            eta = init_eta,
            gamma = gamma,
            q = q,
            Ups = init_Ups
        )

    @staticmethod
    def steady_state_constraint(
        eta: float, 
        gamma: float,
    ) -> float:
        """Return dynamics covariance according to the steady-state constraint."""
        q = (1 - gamma**2) / eta
        
        return q
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self, 
        bel: LoFiBel, 
        x: Float[Array, "input_dim"],
    ) -> Union[Float[Array, "output_dim"], Any]: 
        m = bel.mean
        m_Y = lambda z: self.model_params.emission_mean_function(z, x)
        y_pred = jnp.atleast_1d(m_Y(m))
        
        return y_pred
        
        


# class RebayesLoFi(Rebayes):
#     def __init__(
#         self,
#         model_params: RebayesParams,
#         lofi_params: LoFiParams,
#         method: str,
#         inflation: str='bayesian',
#     ):
#         self.eta = None
#         self.Ups = None
#         self.gamma = None
#         self.q = None
        
#         if method == 'lofi' or method == 'lofi_orth':
#             if method == 'lofi_orth' and lofi_params.diagonal_covariance:
#                 raise NotImplementedError("Orth-LoFi is not yet implemented for diagonal covariance.")
            
#             initial_cov = model_params.initial_covariance
#             self.eta = 1/initial_cov
#             if lofi_params.diagonal_covariance:
#                 self.Ups = jnp.ones((len(model_params.initial_mean), 1)) * self.eta
#             self.gamma = model_params.dynamics_weights
            
#             if not model_params.dynamics_covariance or (lofi_params.steady_state and not lofi_params.diagonal_covariance):
#                 self.q = (1 - self.gamma**2) / self.eta
#             else:
#                 self.q = model_params.dynamics_covariance
#         else:
#             raise ValueError(f"Unknown method {method}.")
        
#         self.method = method
#         if inflation not in ('bayesian', 'simple', 'hybrid'):
#             raise ValueError(f"Unknown inflation method {inflation}.")
#         self.inflation = inflation
#         self.pp_mean = model_params.initial_mean
#         self.nobs, self.obs_noise_var = 0, 0.0
#         self.model_params = model_params
#         self.adaptive_variance = model_params.adaptive_emission_cov
#         self.m, self.sv_threshold, self.steady_state, self.diagonal_covariance = lofi_params
#         self.U0 = jnp.zeros((len(model_params.initial_mean), self.m))
#         self.Sigma0 = jnp.zeros((self.m,))
#         self.alpha = model_params.dynamics_covariance_inflation_factor

#     def init_bel(self):
#         return LoFiBel(
#             pp_mean=self.model_params.initial_mean,
#             mean=self.model_params.initial_mean, basis=self.U0, sigma=self.Sigma0,
#             nobs=self.nobs, obs_noise_var=self.obs_noise_var,
#             eta=self.eta, Ups=self.Ups, gamma=self.gamma, q=self.q,
#         )
    
#     @partial(jit, static_argnums=(0,))
#     def predict_state(self, bel):
#         m0, m, U, Sigma, nobs, obs_noise_var, eta, Ups_pred, gamma = \
#             bel.pp_mean, bel.mean, bel.basis, bel.sigma, bel.nobs, bel.obs_noise_var, bel.eta, bel.Ups, bel.gamma
            
#         if self.method == 'orfit':
#             return bel
#         elif self.diagonal_covariance:
#             m_pred, U_pred, Sigma_pred, Ups_pred = \
#                 _lofi_diagonal_cov_predict(m, U, Sigma, self.gamma, self.q, Ups_pred)
#             m_pred, U_pred, Sigma_pred, Ups_pred, eta_pred = \
#                 _lofi_diagonal_cov_inflate(m0, m_pred, U_pred, Sigma_pred, self.gamma, self.q, eta, Ups_pred, self.alpha, self.inflation)
#         else:
#             m_pred, U_pred, Sigma_pred, eta_pred = \
#                 _lofi_spherical_cov_predict(m, U, Sigma, self.gamma, self.q, eta, self.steady_state)
#             m_pred, U_pred, Sigma_pred, eta_pred = \
#                 _lofi_spherical_cov_inflate(m0, m_pred, U_pred, Sigma_pred, eta_pred, self.alpha, self.inflation)
#         m0 = gamma*m0
        
#         return bel.replace(
#             pp_mean=m0, mean=m_pred, basis=U_pred, sigma=Sigma_pred,
#             nobs=nobs, obs_noise_var=obs_noise_var, eta=eta_pred, Ups=Ups_pred,
#         )

#     @partial(jit, static_argnums=(0,))
#     def predict_obs(self, bel, u):
#         m, U, sigma, obs_noise_var = \
#             bel.mean, bel.basis, bel.sigma, bel.obs_noise_var
#         m_Y = lambda z: self.model_params.emission_mean_function(z, u)
#         y_pred = jnp.atleast_1d(m_Y(m))
#         return y_pred
    
#     @partial(jit, static_argnums=(0,))
#     def predict_obs_cov(self, bel, u):
#         m, U, sigma, obs_noise_var, eta, Ups = \
#             bel.mean, bel.basis, bel.sigma, bel.obs_noise_var, bel.eta, bel.Ups
#         m_Y = lambda z: self.model_params.emission_mean_function(z, u)
#         Cov_Y = lambda z: self.model_params.emission_cov_function(z, u)
        
#         # Predicted mean
#         y_pred = jnp.atleast_1d(m_Y(m))

#         # Predicted covariance
#         H =  _jacrev_2d(m_Y, m)
#         if self.method == 'orfit':
#             Sigma_obs = H @ H.T - (H @ U) @ (H @ U).T
#         else:
#             if self.adaptive_variance:
#                 R = jnp.eye(y_pred.shape[0]) * obs_noise_var
#             else:
#                 R = jnp.atleast_2d(Cov_Y(m))
                
#             if self.diagonal_covariance:
#                 W = U * sigma
#                 G = jnp.linalg.pinv(jnp.eye(W.shape[1]) +  W.T @ (W/Ups))
#                 HW = H/Ups @ W
#                 V_epi = H @ H.T/Ups - (HW @ G) @ (HW).T
#             else:
#                 G = (sigma**2)/(eta * (eta + sigma**2))
#                 HU = H @ U
#                 V_epi = H @ H.T/eta - (G * HU) @ (HU).T
    
#             Sigma_obs = V_epi + R
                    
#         return Sigma_obs

#     @partial(jit, static_argnums=(0,))
#     def update_state(self, bel, u, y):
#         m, U, Sigma, nobs, obs_noise_var, eta_cond, Ups_cond = \
#             bel.mean, bel.basis, bel.sigma, bel.nobs, bel.obs_noise_var, bel.eta, bel.Ups
            
#         if self.method == 'orfit':
#             m_cond, U_cond, Sigma_cond = _orfit_condition_on(
#                 m, U, Sigma, self.model_params.emission_mean_function, u, y, self.sv_threshold
#             )
#         else:
#             nobs, obs_noise_var = _lofi_estimate_noise(
#                 m, self.model_params.emission_mean_function,
#                 u, y, nobs, obs_noise_var, self.adaptive_variance
#             )
#             if self.method == 'lofi':
#                 if self.diagonal_covariance:
#                     m_cond, U_cond, Sigma_cond, Ups_cond = _lofi_diagonal_cov_condition_on(
#                         m, U, Sigma, Ups_cond, self.model_params.emission_mean_function, 
#                         self.model_params.emission_cov_function, u, y, self.sv_threshold, 
#                         self.adaptive_variance, obs_noise_var
#                     )
#                 else:
#                     m_cond, U_cond, Sigma_cond, _ = _lofi_spherical_cov_condition_on(
#                         m, U, Sigma, eta_cond, self.model_params.emission_mean_function, 
#                         self.model_params.emission_cov_function, u, y, self.sv_threshold, 
#                         self.adaptive_variance, obs_noise_var
#                     )
#             elif self.method == 'lofi_orth':
#                 m_cond, U_cond, Sigma_cond = _lofi_orth_condition_on(
#                     m, U, Sigma, eta_cond, self.model_params.emission_mean_function, 
#                     self.model_params.emission_cov_function, u, y, self.sv_threshold, 
#                     self.adaptive_variance, obs_noise_var, nobs
#                 )

#         return bel.replace(
#             mean=m_cond, basis=U_cond, sigma=Sigma_cond, nobs=nobs, 
#             obs_noise_var=obs_noise_var, eta=eta_cond, Ups=Ups_cond
#         )


# def _invert_2x2_block_matrix(M, lr_block_dim):
#     """Invert a 2x2 block matrix. The matrix is assumed to be of the form:
#     [[A, b],
#     [b.T, c]]
#     where A is a diagonal matrix.

#     Args:
#         M (2, 2): 2x2 block matrix.
#         lr_block_dim (int): Dimension of the lower right block.
        
#     Returns:
#         (2, 2): Inverse of the 2x2 block matrix.
#     """
#     m, n = M.shape
#     A = M[:m-lr_block_dim, :n-lr_block_dim]
#     B = M[:m-lr_block_dim, n-lr_block_dim:]
#     D = M[m-lr_block_dim:, n-lr_block_dim:]
#     a = 1/jnp.diag(A)
#     K_inv = jnp.linalg.inv(D - (a*B.T) @ B)

#     B_inv = - (a * B.T).T @ K_inv
#     A_inv = jnp.diag(a) + (a * B.T).T @ K_inv @ (a * B.T)
#     C_inv = -K_inv @ (a * B.T)
#     D_inv = K_inv

#     return jnp.block([[A_inv, B_inv], [C_inv, D_inv]])


# def _lofi_orth_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False, obs_noise_var=1.0, key=0):
#     """Condition step of the low-rank filter algorithm based on orthogonal SVD method.

#     Args:
#         m (D_hid,): Prior mean.
#         U (D_hid, D_mem,): Prior basis.
#         Sigma (D_mem,): Prior singular values.
#         eta (float): Prior precision.
#         y_cond_mean (Callable): Conditional emission mean function.
#         y_cond_cov (Callable): Conditional emission covariance function.
#         x (D_in,): Control input.
#         y (D_obs,): Emission.
#         sv_threshold (float): Threshold for singular values.
#         adaptive_variance (bool): Whether to use adaptive variance.
#         key (int): Random key.

#     Returns:
#         m_cond (D_hid,): Posterior mean.
#         U_cond (D_hid, D_mem,): Posterior basis.
#         Sigma_cond (D_mem,): Posterior singular values.
#     """
#     if isinstance(key, int):
#         key = jr.PRNGKey(key)
    
#     m_Y = lambda w: y_cond_mean(w, x)
#     Cov_Y = lambda w: y_cond_cov(w, x)
    
#     yhat = jnp.atleast_1d(m_Y(m))
#     if adaptive_variance:
#         R = jnp.eye(yhat.shape[0]) * obs_noise_var
#     else:
#         R = jnp.atleast_2d(Cov_Y(m))
#     L = jnp.linalg.cholesky(R)
#     A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
#     H = _jacrev_2d(m_Y, m)
#     W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])
#     S = eta*jnp.eye(W_tilde.shape[1]) + W_tilde.T @ W_tilde
#     K = (H.T @ A) @ A.T - W_tilde @ (_invert_2x2_block_matrix(S, yhat.shape[0]) @ (W_tilde.T @ ((H.T @ A) @ A.T)))

#     m_cond = m + K/eta @ (y - yhat)

#     def _update_basis(carry, i):
#         U, Sigma = carry
#         U_tilde = (H.T - U @ (U.T @ H.T)) @ A
#         v = U_tilde[:, i]
#         u = _normalize(v)
#         U_cond = jnp.where(
#             Sigma.min() < u @ v, 
#             jnp.where(sv_threshold < u @ v, U.at[:, Sigma.argmin()].set(u), U),
#             U
#         )
#         Sigma_cond = jnp.where(
#             Sigma.min() < u @ v,
#             jnp.where(sv_threshold < u @ v, Sigma.at[Sigma.argmin()].set(u.T @ v), Sigma),
#             Sigma,
#         )
#         return (U_cond, Sigma_cond), (U_cond, Sigma_cond)

#     perm = jr.permutation(key, yhat.shape[0])
#     (U_cond, Sigma_cond), _ = scan(_update_basis, (U, Sigma), perm)

#     return m_cond, U_cond, Sigma_cond


# def _lofi_spherical_cov_condition_on(m, U, Sigma, eta, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False, obs_noise_var=1.0):
#     """Condition step of the low-rank filter with adaptive observation variance.

#     Args:
#         m (D_hid,): Prior mean.
#         U (D_hid, D_mem,): Prior basis.
#         Sigma (D_mem,): Prior singular values.
#         eta (float): Prior precision. 
#         y_cond_mean (Callable): Conditional emission mean function.
#         y_cond_cov (Callable): Conditional emission covariance function.
#         x (D_in,): Control input.
#         y (D_obs,): Emission.
#         sv_threshold (float): Threshold for singular values.
#         adaptive_variance (bool): Whether to use adaptive variance.

#     Returns:
#         m_cond (D_hid,): Posterior mean.
#         U_cond (D_hid, D_mem,): Posterior basis.
#         Sigma_cond (D_mem,): Posterior singular values.
#     """
#     m_Y = lambda w: y_cond_mean(w, x)
#     Cov_Y = lambda w: y_cond_cov(w, x)
    
#     yhat = jnp.atleast_1d(m_Y(m))
#     if adaptive_variance:
#         R = jnp.eye(yhat.shape[0]) * obs_noise_var
#     else:
#         R = jnp.atleast_2d(Cov_Y(m))
#     L = jnp.linalg.cholesky(R)
#     A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
#     H = _jacrev_2d(m_Y, m)
#     W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])

#     # Update the U matrix
#     u, lamb, _ = jnp.linalg.svd(W_tilde, full_matrices=False)

#     D = (lamb**2)/(eta**2 + eta * lamb**2)
#     K = (H.T @ A) @ A.T/eta - (D * u) @ (u.T @ ((H.T @ A) @ A.T))

#     U_cond = u[:, :U.shape[1]]
#     Sigma_cond = lamb[:U.shape[1]]

#     m_cond = m + K @ (y - yhat)

#     return m_cond, U_cond, Sigma_cond, eta


# def _lofi_diagonal_cov_condition_on(m, U, Sigma, Ups, y_cond_mean, y_cond_cov, x, y, sv_threshold, adaptive_variance=False, obs_noise_var=1.0):
#     """Condition step of the low-rank filter with adaptive observation variance.

#     Args:
#         m (D_hid,): Prior mean.
#         U (D_hid, D_mem,): Prior basis.
#         Sigma (D_mem,): Prior singular values.
#         Ups (D_hid): Prior precision. 
#         y_cond_mean (Callable): Conditional emission mean function.
#         y_cond_cov (Callable): Conditional emission covariance function.
#         x (D_in,): Control input.
#         y (D_obs,): Emission.
#         sv_threshold (float): Threshold for singular values.
#         adaptive_variance (bool): Whether to use adaptive variance.

#     Returns:
#         m_cond (D_hid,): Posterior mean.
#         U_cond (D_hid, D_mem,): Posterior basis.
#         Sigma_cond (D_mem,): Posterior singular values.
#     """
#     m_Y = lambda w: y_cond_mean(w, x)
#     Cov_Y = lambda w: y_cond_cov(w, x)
    
#     yhat = jnp.atleast_1d(m_Y(m))
#     if adaptive_variance:
#         R = jnp.eye(yhat.shape[0]) * obs_noise_var
#     else:
#         R = jnp.atleast_2d(Cov_Y(m))
#     L = jnp.linalg.cholesky(R)
#     A = jnp.linalg.lstsq(L, jnp.eye(L.shape[0]))[0].T
#     H = _jacrev_2d(m_Y, m)
#     W_tilde = jnp.hstack([Sigma * U, (H.T @ A).reshape(U.shape[0], -1)])
    
#     # Update the U matrix
#     u, lamb, _ = jnp.linalg.svd(W_tilde, full_matrices=False)
#     U_cond, U_extra = u[:, :U.shape[1]], u[:, U.shape[1]:]
#     Sigma_cond, Sigma_extra = lamb[:U.shape[1]], lamb[U.shape[1]:]
#     W_extra = Sigma_extra * U_extra
#     Ups_cond = Ups + jnp.einsum('ij,ij->i', W_extra, W_extra)[:, jnp.newaxis]
    
#     G = jnp.linalg.pinv(jnp.eye(W_tilde.shape[1]) + W_tilde.T @ (W_tilde/Ups))
#     K = (H.T @ A) @ A.T/Ups - (W_tilde/Ups @ G) @ ((W_tilde/Ups).T @ (H.T @ A) @ A.T)
#     m_cond = m + K @ (y - yhat)
    
#     return m_cond, U_cond, Sigma_cond, Ups_cond


# def _lofi_estimate_noise(
#     m: Float[Array, "state_dim"],
#     y_cond_mean: Callable,
#     x: Float[Array, "input_dim"],
#     y: Float[Array, "obs_dim"],
#     nobs: int,
#     obs_noise_var: float,
#     adaptive_variance: bool = False
# ) -> Tuple[int, float]:
#     """Estimate observation noise based on empirical residual errors.

#     Args:
#         m (D_hid,): Prior mean.
#         y_cond_mean (Callable): Conditional emission mean function.
#         x (D_in,): Control input.
#         y (D_obs,): Emission.
#         nobs (int): Number of observations seen so far.
#         obs_noise_var (float): Current estimate of observation noise.
#         adaptive_variance (bool): Whether to use adaptive variance.

#     Returns:
#         nobs (int): Updated number of observations seen so far.
#         obs_noise_var (float): Updated estimate of observation noise.
#     """
#     if not adaptive_variance:
#         return 0, 0.0

#     m_Y = lambda w: y_cond_mean(w, x)
#     yhat = jnp.atleast_1d(m_Y(m))
    
#     sqerr = ((yhat - y).T @ (yhat - y)).squeeze() / yhat.shape[0]
#     nobs += 1
#     obs_noise_var = jnp.max(jnp.array([1e-6, obs_noise_var + 1/nobs * (sqerr - obs_noise_var)]))

#     return nobs, obs_noise_var


# def _lofi_spherical_cov_inflate(m0, m, U, Sigma, eta, alpha, inflation='bayesian'):
#     """Inflate the diagonal covariance matrix.

#     Args:
#         Ups (D_hid,): Prior diagonal covariance.
#         alpha (float): Covariance inflation factor.

#     Returns:
#         Ups (D_hid,): Inflated diagonal covariance.
#     """
#     Sigma_pred = Sigma/jnp.sqrt(1+alpha)
#     U_pred = U
#     W_pred = U_pred * Sigma_pred
    
#     if inflation == 'bayesian':
#         eta_pred = eta
#         G = jnp.linalg.pinv(jnp.eye(W_pred.shape[1]) +  (W_pred.T @ (W_pred/eta_pred)))
#         e = (m0 - m)
#         K = e - ((W_pred/eta_pred) @ G) @ (W_pred.T @ e)
#         m_pred = m + alpha/(1+alpha) * K.ravel()
#     elif inflation == 'simple':
#         eta_pred = eta/(1+alpha)
#         m_pred = m
#     elif inflation == 'hybrid':
#         eta_pred = eta
#         m_pred = m
    
#     return m_pred, U_pred, Sigma_pred, eta_pred


# def _lofi_spherical_cov_predict(m, U, Sigma, gamma, q, eta, steady_state=False):
#     """Predict step of the low-rank filter algorithm.

#     Args:
#         m0 (D_hid,): Initial mean.
#         m (D_hid,): Prior mean.
#         U (D_hid, D_mem,): Prior basis.
#         Sigma (D_mem,): Prior singluar values.
#         gamma (float): Dynamics decay factor.
#         q (float): Dynamics noise factor.
#         eta (float): Prior precision.
#         alpha (float): Covariance inflation factor.
#         steady_state (bool): Whether to use steady-state dynamics.

#     Returns:
#         m_pred (D_hid,): Predicted mean.
#         Sigma_pred (D_mem,): Predicted singular values.
#         eta_pred (float): Predicted precision.
#     """
#     # Mean prediction
#     m_pred = gamma*m

#     # Covariance prediction
#     U_pred = U
    
#     if steady_state:
#         eta_pred = eta
#         Sigma_pred = jnp.sqrt(
#             (gamma**2 * Sigma**2) /
#             (1 + q*Sigma**2)
#         )
#     else:
#         eta_pred = eta/(gamma**2 + q*eta)
#         Sigma_pred = jnp.sqrt(
#             (gamma**2 * Sigma**2) /
#             ((gamma**2 + q*eta) * (gamma**2 + q*eta + q*Sigma**2))
#         )

#     return m_pred, U_pred, Sigma_pred, eta_pred


# def _lofi_diagonal_cov_inflate(m0, m, U, Sigma, gamma, q, eta, Ups, alpha, inflation='bayesian'):
#     """Inflate the diagonal covariance matrix.

#     Args:
#         Ups (D_hid,): Prior diagonal covariance.
#         alpha (float): Covariance inflation factor.

#     Returns:
#         Ups (D_hid,): Inflated diagonal covariance.
#     """
#     W = U * Sigma
#     eta_pred = eta/(gamma**2 + q*eta)
    
#     if inflation == 'bayesian':
#         W_pred = W/jnp.sqrt(1+alpha)
#         Ups_pred = Ups/(1+alpha) + alpha*eta/(1+alpha)
#         G = jnp.linalg.pinv(jnp.eye(W.shape[1]) +  (W_pred.T @ (W_pred/Ups_pred)))
#         e = (m0 - m)
#         K = 1/Ups_pred.ravel() * (e - (W_pred @ G) @ ((W_pred/Ups_pred).T @ e))
#         m_pred = m + alpha*eta/(1+alpha) * K
#     elif inflation == 'simple':
#         W_pred = W/jnp.sqrt(1+alpha)
#         Ups_pred = Ups/(1+alpha)
#         m_pred = m
#     elif inflation == 'hybrid':
#         W_pred = W/jnp.sqrt(1+alpha)
#         Ups_pred = Ups/(1+alpha) + alpha*eta/(1+alpha)
#         m_pred = m
#     U_pred, Sigma_pred, _ = jnp.linalg.svd(W_pred, full_matrices=False)
    
#     return m_pred, U_pred, Sigma_pred, Ups_pred, eta_pred


# def _lofi_diagonal_cov_predict(m, U, Sigma, gamma, q, Ups, steady_state=False):
#     """Predict step of the generalized low-rank filter algorithm.

#     Args:
#         m0 (D_hid,): Initial mean.
#         m (D_hid,): Prior mean.
#         U (D_hid, D_mem,): Prior basis.
#         Sigma (D_mem,): Prior singluar values.
#         gamma (float): Dynamics decay factor.
#         q (float): Dynamics noise factor.
#         eta (float): Prior precision.
#         Ups (D_hid,): Prior diagonal covariance.
#         alpha (float): Covariance inflation factor.

#     Returns:
#         m_pred (D_hid,): Predicted mean.
#         U_pred (D_hid, D_mem,): Predicted basis.
#         Sigma_pred (D_mem,): Predicted singular values.
#         eta_pred (float): Predicted precision.
#     """
#     # Mean prediction
#     W = U * Sigma
#     m_pred = gamma*m

#     # Covariance prediction
#     Ups_pred = 1/(gamma**2/Ups + q)
#     C = jnp.linalg.pinv(jnp.eye(W.shape[1]) + q*W.T @ (W*(Ups_pred/Ups)))
#     W_pred = gamma*(Ups_pred/Ups)*W @ jnp.linalg.cholesky(C)
#     U_pred, Sigma_pred, _ = jnp.linalg.svd(W_pred, full_matrices=False)
    
#     return m_pred, U_pred, Sigma_pred, Ups_pred
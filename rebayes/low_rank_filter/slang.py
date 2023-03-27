from functools import partial
from typing import Any

import chex
from jax import jit, grad
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalDiag as MVND,
    MultivariateNormalFullCovariance as MVN,
)

from rebayes.base import (
    FnStateAndInputToEmission,
    FnStateAndInputToEmission2,
    Rebayes,
)


@chex.dataclass
class SLANGBel:
    mean: Float[Array, "state_dim"]
    cov_diag: Float[Array, "state_dim"]
    cov_lr: Float[Array, "state_dim memory_size"]
    step: int = 0


@chex.dataclass
class SLANGParams:
    """Lightweight container for SLANG parameters.
    """
    initial_mean: Float[Array, "state_dim"]
    initial_cov_diag: Float[Array, "state_dim"]
    initial_cov_lr: Float[Array, "state_dim memory_size"]
    emission_mean_function: FnStateAndInputToEmission
    emission_cov_function: FnStateAndInputToEmission2
    lamb: float
    alpha: float
    beta: float
    batch_size: int
    n_train: int
    n_eig: int = 10
    likelihood_dist: Any = lambda m, sigma : MVN(loc=m, covariance_matrix=sigma)


class SLANG(Rebayes):
    def __init__(
        self,
        params: SLANGParams,
    ):
        self.params = params
        self.log_lik = lambda mu, x, y: \
            self.params.likelihood_dist(
                self.params.emission_mean_function(mu, x), 
                self.params.emission_cov_function(mu, x)
            ).log_prob(y)
        self.grad_log_lik = jit(grad(self.log_lik, argnums=(0)))
    
    def init_bel(
        self
    ) -> SLANGBel:
        bel = SLANGBel(
            mean=self.params.initial_mean,
            cov_diag=self.params.initial_cov_diag,
            cov_lr=self.params.initial_cov_lr,
        )
        
        return bel
    
    @partial(jit, static_argnums=(0,))
    def predict_obs(
        self,
        bel: SLANGBel,
        x: Float[Array, "input_dim"]
    ) -> Float[Array, "output_dim"]:
        y_pred = self.params.emission_mean_function(bel.mean, x)
        
        return y_pred
    
    @partial(jit, static_argnums=(0,))
    def update_state(
        self,
        bel: SLANGBel,
        x: Float[Array, "input_dim"],
        y: Float[Array, "output_dim"],
    ) -> SLANGBel:
        """Update the belief state given an input and output.
        """
        m, U, d = bel.mean, bel.cov_lr, bel.cov_diag
        D, L = U.shape
        alpha, beta, lamb, n_eig = \
            self.params.alpha, self.params.beta, self.params.lamb, self.params.n_eig
        
        theta = self._fast_sample(bel)
        g = self.grad_log_lik(theta, x, y).reshape((D, -1))
        V = self._fast_eig(bel, g, beta, n_eig)
        diag_corr = (1-beta) * (U**2).sum(axis=1) + beta * (g**2).ravel() - (V**2).sum(axis=1)
        
        U_post = V
        d_post = (1-beta) * d + diag_corr + lamb * jnp.ones(D)
        
        ghat = g + lamb*m.reshape((D, -1))
        m_post = m - alpha*self._fast_inverse(SLANGBel(mean=m, cov_lr=U_post, cov_diag=d_post), ghat)
        
        # Construct the new belief
        bel = SLANGBel(
            mean=m_post,
            cov_lr=U_post,
            cov_diag=d_post,
        )
        
        return bel
    
    def _fast_sample(
        self,
        bel: SLANGBel,
    ):
        mu, U, d = bel.mean, bel.cov_lr, bel.cov_diag
        key = jr.PRNGKey(bel.step)

        D, L = U.shape
        eps = MVND(loc=jnp.zeros(D,), scale_diag=jnp.ones(D,)).sample(seed=key)
        dd = 1/jnp.sqrt(d).reshape((d.shape[0], 1))
        V = U * dd
        A = jnp.linalg.cholesky(V.T @ V)
        B = jnp.linalg.cholesky(jnp.eye(L) + V.T @ V)
        C = jnp.linalg.pinv(A.T) @ (B - jnp.eye(L)) @ jnp.linalg.pinv(A)
        K = jnp.linalg.pinv(jnp.linalg.pinv(C) + V.T @ V)
        y = dd.ravel() * eps - (V * dd) @ K @ (V.T @ eps)

        return mu + y
    
    def _fast_eig(
        self,
        bel,
        g,
        beta,
        n_iter,
    ):
        key = jr.PRNGKey(bel.step+1)
        U = bel.cov_lr
        D, L = U.shape
        K = L + 2
        Q = jr.uniform(key, shape=(D, K), minval=-1.0, maxval=1.0)
        
        def _orth_step(carry, i):
            Q = carry
            AQ = (1-beta) * U @ (U.T@Q) + beta * g @ (g.T @ Q)
            Q_orth, _ = jnp.linalg.qr(AQ)

            return Q_orth, Q_orth
        
        Q_orth, _ = scan(_orth_step, Q, jnp.arange(n_iter))
        V, *_ = jnp.linalg.svd(Q_orth, full_matrices=False)
        V = V[:, :L]

        return V

    def _fast_inverse(
        self,
        bel,
        g,
    ):
        U, d = bel.cov_lr, bel.cov_diag
        _, L = U.shape
        dinv = (1/d).reshape((d.shape[0], 1))
        A = jnp.linalg.pinv(jnp.eye(L) + U.T @ (U*dinv))
        y = dinv.ravel() * g.ravel() - ((U*dinv) @ A) @ ((U*dinv).T @ g).ravel()

        return y
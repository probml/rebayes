"""
Implementation of the Recursive Variational Gaussian Approximation
(R-VGA) and the Limited-memory Recursive Variational Gaussian Approximation
(LR-VGA) [1] algorithms for sequential estimation.

[1] Lambert, M., Bonnabel, S., & Bach, F. (2021, December).
The limited-memory recursive variational Gaussian approximation (L-RVGA).
Retrieved from https://hal.inria.fr/hal-03501920
"""

import jax
import jax.numpy as jnp
from typing import Callable
from functools import partial
from flax import struct
from jaxtyping import Array, Float
from rebayes.base import Rebayes
from jax.flatten_util import ravel_pytree


# Homoskedastic case (we estimate sigma at a warmup stage)
@struct.dataclass
class LRVGAState:
    key: jax.random.PRNGKey
    mu: Float[Array, "dim_params"]
    W: Float[Array, "dim_params dim_subspace"]
    Psi: Float[Array, "dim_params"]
    sigma: float = 1.0
    step: int = 0

    @property
    def mean(self):
        return self.mu

    @property
    def cov(self):
        precision_matrix = self.W @ self.W.T + jnp.diag(self.Psi)
        return jnp.linalg.inv(precision_matrix)


def _init_lowrank(key, eps, sigma2, dim_params, dim_latent):
    """
    Initialise low-rank variational Gaussian approximation
    components W adn Ps sigma2,
    See §5.1.2 for initialisation details
    """
    psi0 = (1 - eps) / sigma2

    w0 = jnp.sqrt((eps * dim_params) / (dim_latent * sigma2))

    W_init = jax.random.normal(key, (dim_params, dim_latent))
    W_init = W_init / jnp.linalg.norm(W_init, axis=0) * w0
    Psi_init = jnp.ones(dim_params) * psi0
    return W_init, Psi_init


def init_lrvga(key, model, X_init, dim_rank, std, eps, sigma2):
    key_W, key_mu, key_carry = jax.random.split(key, 3)

    mu_init = model.init(key_mu, X_init)
    mu_init, reconstruct_fn = ravel_pytree(mu_init)
    mu_init = jnp.array(mu_init)
    dim_params = len(mu_init)

    W_init, Psi_init = _init_lowrank(key_W, eps, sigma2, dim_params, dim_rank)


    bel_init = LRVGAState(
        key=key_carry,
        mu=mu_init,
        W=W_init,
        Psi=Psi_init,
        sigma=std,
    )

    return bel_init, reconstruct_fn

class LRVGA(Rebayes):
    """
    Limited-memory recursive variational Gaussian approximation (LRVGA)
    for a homoskedastic Gaussian model with known mean
    """
    def __init__(
            self,
            fwd_link: Callable,
            log_prob: Callable,
            alpha: float = 1.0,
            beta: float = 1.0,
            n_outer: int = 3,
            n_inner: int = 3,
            n_samples: int = 6,
    ):
        self.fwd_link = fwd_link
        self.log_prob = log_prob
        self.alpha = alpha
        self.beta = beta
        self.n_outer = n_outer # Solve implicit scheme
        self.n_inner = n_inner # Solve for FA approximation
        self.n_samples = n_samples
        self.grad_log_prob = jax.grad(log_prob, argnums=0)

    @staticmethod
    def _sample_lr_params(key, bel):
        """
        Sample parameters from a low-rank variational Gaussian approximation.
        This implementation avoids the explicit construction of the
        (D x D) covariance matrix.

        We take s ~ N(0, W W^T + Psi I)

        Implementation based on §4.2.2 of the L-RVGA paper.

        TODO(?): refactor code into jax.vmap. (It faster?)
        """
        key_x, key_eps = jax.random.split(key)
        dim_full, dim_latent = bel.W.shape
        Psi_inv = 1 / bel.Psi

        eps_sample = jax.random.normal(key_eps, (dim_latent,))
        x_sample = jax.random.normal(key_x, (dim_full,)) * jnp.sqrt(Psi_inv)

        I_full = jnp.eye(dim_full)
        I_latent = jnp.eye(dim_latent)
        # M = I + W^T Psi^{-1} W
        M = I_latent + jnp.einsum("ji,j,jk->ik", bel.W, Psi_inv, bel.W)
        # L = Psi^{-1} W^T M^{-1}
        L_tr = jnp.linalg.solve(M.T, jnp.einsum("i,ij->ji", Psi_inv, bel.W))

        # samples = (I - LW^T)x + Le
        term1 = I_full - jnp.einsum("ji,kj->ik", L_tr, bel.W)
        x_transform = jnp.einsum("ij,j->i", term1, x_sample)
        eps_transform = jnp.einsum("ji,j->i", L_tr, eps_sample)
        samples = x_transform + eps_transform
        return samples + bel.mu

    @staticmethod
    def _get_coef(params, bel, x, fwd_link):
        c, std = jax.jacfwd(fwd_link, has_aux=True)(params, bel, x)
        std = jnp.sqrt(std)
        return c * std

    @partial(jax.vmap, in_axes=(None, 0, None, None, None))
    def _sample_grad_expected_log_prob(self, key, bel, x, y):
        """
        E[∇ logp(y|x,θ)]
        """
        mu_sample = self._sample_lr_params(key, bel)
        grads = self.grad_log_prob(mu_sample, bel, x, y)
        return grads

    @partial(jax.vmap, in_axes=(None, 0, None, None))
    def _sample_cov_coeffs(self, key, x, bel):
        params = self._sample_lr_params(key, bel)
        coef = self._get_coef(params, bel, x, self.fwd_link)
        return coef

    def _fa_approx_step(
        self,
        x: Float[Array, "dim_params"],
        bel: LRVGAState,
        bel_prev: LRVGAState,
    ) -> LRVGAState:
        """
        Factor Analysis (FA) approximation to the low-rank (W)
        and diagonal (Psi) matrices.
        """
        # Load data
        W_prev, Psi_prev = bel_prev.W, bel_prev.Psi
        W, Psi = bel.W, bel.Psi

        # Initialise basic transformations
        _, dim_latent = W.shape
        I = jnp.eye(dim_latent)
        Psi_inv = 1 / Psi

        # Construct helper matrices
        M = I + jnp.einsum("ij,i,ik->jk", W, Psi_inv, W)
        M_inv = jnp.linalg.inv(M)
        V_beta = jnp.einsum("...i,...j,j,jk->ik", x, x, Psi_inv, W)
        V_alpha = (
            jnp.einsum("ij,kj,k,kl->il", W_prev, W_prev, Psi_inv, W) +
            jnp.einsum("i,i,ij->ij", Psi_prev, Psi_inv, W)
        )
        V = self.beta * V_beta + self.alpha * V_alpha
        # Value_update
        # (return transpose of W_solve -- avoid extra transpose op)
        W_solve = I + jnp.einsum("ij,kj,k,kl->li", M_inv, W, Psi_inv, V)
        W = jnp.linalg.solve(W_solve, V.T).T
        Psi = (
            self.beta * jnp.einsum("...i,...i->i", x, x) +
            self.alpha * jnp.einsum("ij,ij->i", W_prev, W_prev) + 
            self.alpha * Psi_prev -
            jnp.einsum("ij,jk,ik->i", W, M_inv, V)
        )

        new_bel = bel.replace(
            mu=bel.mu,
            W=W,
            Psi=Psi
        )
        return new_bel

    def _mu_update(
        self,
        key,
        x: Float[Array, "dim_obs"],
        y: float,
        bel_prev: LRVGAState,
        bel: LRVGAState,
    ) -> Float[Array, "dim_obs"]:
        """
        Obtain gain matrix-vector multiplication for the mean update.

        TODO: Optimise for lower compilation time:
            1. Refactor sample_predictions
            2. Refactor sample_grad_expected_log_prob
        TODO: Rewrite the V term using the Woodbury matrix identity
        """
        W = bel.W
        dim_full, _ = W.shape
        I = jnp.eye(dim_full)

        keys_grad = jax.random.split(key, self.n_samples)

        V = W @ W.T + bel.Psi * I
        exp_grads_log_prob = self._sample_grad_expected_log_prob(keys_grad, bel_prev, x, y).mean(axis=0)
        gain = jnp.linalg.solve(V, exp_grads_log_prob)
        return gain

    def _sample_half_fisher(self, key, x, bel):
        """
        Estimate X such that
            X X^T ~ E_{q(θ)}[E_{y}[∇^2 log p(y|x,θ)]]
        """
        keys = jax.random.split(key, self.n_samples)
        coeffs = self._sample_cov_coeffs(keys, x, bel) / jnp.sqrt(self.n_samples)
        # XXtr = jnp.einsum("nji,njk->ik", coeffs, coeffs) / num_samples
        return coeffs

    def _step_lrvga(self, bel, key, x, y):
        """
        Iterated RVGA (§4.2.1). We omit the second iteration of the covariance matrix
        """
        key_fisher, key_est, key_mu_final = jax.random.split(key, 3)

        X = self._sample_half_fisher(key_fisher, x, bel)
        def fa_partial(_, new_bel):
            new_bel = self._fa_approx_step(X, new_bel, bel)
            return new_bel

        # Algorithm 1 in §3.2 of L-RVGA suggests that 1 to 3 loops may be enough in
        # the inner (fa-update) loop (See comments in Algorithm 1)

        # Estimate hat{P} (Eq 36 - 1)
        bel_update = jax.lax.fori_loop(0, self.n_inner, fa_partial, bel)
        # First mu update (Eq 36 - 2)
        mu_add = self._mu_update(key_est, x, y, bel, bel_update)
        mu_new = bel.mu + mu_add
        bel_update = bel_update.replace(mu=mu_new)
        # Second mu update (Eq 36 - 4)
        # we use the updated bel to estimate the gradient
        mu_add = self._mu_update(key_mu_final, x, y, bel_update, bel_update)
        mu_new = bel.mu + mu_add
        bel_update = bel_update.replace(mu=mu_new, step=bel_update.step + 1)
        return bel_update

    def init_bel(self):
        raise NotImplementedError
        # TODO: Implement initialisation.
        # TODO: Modify base class to allow for initialisation kwargs
        if self.key is None:
            raise ValueError("Must provide a key to initialise belief")

    def predict_obs(self, bel, X):
        yhat, var = self.fwd_link(bel.mean, bel, X)
        # return Gaussian(mean=yhat, cov=var)
        return yhat

    def predict_state(self, bel):
        """
        L-RVGA doesn't have a closed-form prediction for the state
        (check)
        """
        return bel

    def update_state(self, bel, Xt, yt):
        key = jax.random.fold_in(bel.key, bel.step)

        def _step(i, bel):
            key_i = jax.random.fold_in(key, i)
            bel = self._step_lrvga(bel, key_i, Xt, yt)
            return bel
        bel = jax.lax.fori_loop(0, self.n_outer, _step, bel)
        return bel

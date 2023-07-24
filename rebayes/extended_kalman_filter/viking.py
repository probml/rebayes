import chex
import jax
import jax.numpy as jnp
from jax.lax import scan


@chex.dataclass
class VikingBel:
    mean: chex.Array
    cov: chex.Array
    dynamics_noise_mean: chex.Array
    dynamics_noise_cov: chex.Array
    emission_noise_mean: chex.Array
    emission_noise_cov: chex.Array


class RebayesViking:
    def __init__(
        self,
        dynamics_weights, # K
        initial_dynamics_noise_mean, # b_hat
        initial_dynamics_noise_cov,  # Sigma
        initial_emission_noise_mean, # a_hat
        initial_emission_noise_cov,  # s
        n_iter,
        dynamics_noise_transition_cov=0.0, # rho_b
        emission_noise_transition_cov=0.0, # rho_a
        learn_dynamics_noise_cov=True,
        learn_emission_noise_cov=True,
    ):
        self.K = dynamics_weights
        self.b_hat0 = initial_dynamics_noise_mean
        self.Sigma0 = initial_dynamics_noise_cov
        self.a_hat0 = initial_emission_noise_mean
        self.s0 = initial_emission_noise_cov
        self.n_iter = n_iter
        self.rho_b = dynamics_noise_transition_cov
        self.rho_a = emission_noise_transition_cov
        self.learn_dynamics_noise_cov = learn_dynamics_noise_cov
        self.learn_emission_noise_cov = learn_emission_noise_cov

    def init_bel(
        self,
        initial_mean,
        initial_covariance,
    ) -> VikingBel:
        return VikingBel(
            mean=initial_mean,
            cov=initial_covariance,
            dynamics_noise_mean=self.b_hat0,
            dynamics_noise_cov=self.Sigma0,
            emission_noise_mean=self.a_hat0,
            emission_noise_cov=self.s0,
        )

    def update_bel(
        self,
        bel,
        x,
        y,
    ):
        mean, cov, b_hat, Sigma, a_hat, s = \
            bel.mean, bel.cov, bel.dynamics_noise_mean, bel.dynamics_noise_cov, \
            bel.emission_noise_mean, bel.emission_noise_cov
        a_hat_cond = a_hat
        s_cond = s + self.rho_a
        b_hat_cond = b_hat
        Sigma_cond = Sigma + self.rho_b
        y_pred = mean @ x

        f = jnp.exp(b_hat_cond)
        C_inv = jnp.linalg.pinv(self.K @ cov @ self.K.T + f)
        # Quadrature approximation to the expectation
        A = C_inv - f * Sigma_cond * C_inv @ C_inv / 2 + \
            f**2 * Sigma_cond * C_inv @ C_inv @ C_inv
        A_inv = jnp.linalg.pinv((A+A.T)/2)

        # Update the mean and covariance
        v = jnp.exp(a_hat_cond - s_cond/2)
        cov_cond = A_inv - (A_inv @ x)[:, None] @ x[None, :] \
            @ (A_inv / (x @ A_inv @ x + v))
        mean_cond = self.K @ mean + A_inv @ x / (x @ A_inv @ x + v) \
            * (y - (self.K @ mean) @ x)

        # Update a_hat and s
        if self.learn_emission_noise_cov:
            c = (y - mean_cond @ x)**2 + x @ cov_cond @ x
            s_cond = 1/(1/(s + self.rho_a) + 
                        0.5*c*jnp.exp(-a_hat_cond + (s - self.rho_a)/2))
            s_cond = jax.lax.max(s_cond, s - self.rho_a)
            M = 100*self.rho_a
            diff = 1/2*(1/(s + self.rho_a) + c/2 * 
                        jnp.exp(-a_hat_cond + s_cond/2 + M)) * \
                        (c * jnp.exp(-a_hat_cond + s_cond/2) - 1)
            a_hat_cond = a_hat_cond + jax.lax.max(jax.lax.min(diff, M), -M)
        
        # Update b_hat and Sigma
        if self.learn_dynamics_noise_cov:
            d = mean.shape[0]
            mean_term = mean_cond - self.K @ mean
            B = cov_cond + jnp.outer(mean_term, mean_term)
            C_inv = jnp.linalg.pinv(self.K @ cov @ self.K.T + f)
            g = jnp.sum(jnp.diag(C_inv @ (jnp.eye(d) - B @ C_inv)))*f

            # Approximation and no upper bound
            C_inv = jnp.linalg.pinv(self.K @ cov @ self.K.T + f)
            H = jnp.sum(jnp.diag(C_inv @ (jnp.eye(d) - B @ C_inv)))*f + \
                2*jnp.sum(jnp.diag(C_inv @ C_inv @ (B @ C_inv - jnp.eye(d)/2)))*f**2
            
            # Update b_hat and Sigma
            Sigma_cond = 1/(1/(Sigma + self.rho_b) + H/2)
            b_hat_cond = b_hat_cond - Sigma_cond*g/2

        bel_cond = VikingBel(
            mean=mean_cond,
            cov=cov_cond,
            dynamics_noise_mean=b_hat_cond,
            dynamics_noise_cov=Sigma_cond,
            emission_noise_mean=a_hat_cond,
            emission_noise_cov=s_cond,
        )

        return bel_cond
    
    def scan(
        self,
        initial_mean,
        initial_covariance,
        X,
        Y,
    ):
        X, Y = jnp.array(X), jnp.array(Y)
        num_timesteps = X.shape[0]
        bel = self.init_bel(initial_mean, initial_covariance)
        def step(bel, t):
            x, y = X[t], Y[t]
            def _step(bel, _):
                bel = self.update_bel(bel, x, y)
                return bel, None
            bel, _ = scan(_step, bel, jnp.arange(self.n_iter))
            return bel, None
        
        bel, _ = scan(step, bel, jnp.arange(num_timesteps))

        return bel

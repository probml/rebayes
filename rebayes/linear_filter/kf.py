import jax
import chex
import jax.numpy as jnp
from rebayes.base import Rebayes
from typing import NamedTuple
from functools import partial

@chex.dataclass
class KFBel:
    mean: chex.Array
    cov: chex.Array
    obs_noise_var: float=None


class KalmanFilter(Rebayes):
    def __init__(
        self,
        transition_matrix,
        observation_matrix,
        system_noise_var,
        obs_noise_var,
    ):
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.system_noise_var = system_noise_var
        self.obs_noise_var = obs_noise_var

    def get_trans_mat_of(self, t: int):
        if callable(self.transition_matrix):
            return self.transition_matrix(t)
        else:            
            return self.transition_matrix

    def get_obs_mat_of(self, t: int):
        if callable(self.observation_matrix):
            return self.observation_matrix(t)
        else:
            return self.observation_matrix
        
    def get_system_noise_of(self, t: int):
        if callable(self.system_noise):
            return self.system_noise(t)
        else:
            return self.system_noise

    def get_observation_noise_of(self, t: int):
        if callable(self.observation_noise):
            return self.observation_noise(t)
        else:
            return self.observation_noise


    def init_bel(self, Xinit=None, Yinit=None):
        bel = KFBel(
            mean=self.params.initial_mean,
            cov=self.params.initial_covariance,
            obs_noise_var=self.params.obs_noise_var
        )
        return bel
    
    @partial(jax.jit, static_argnums=(0,))
    def predict_state(self, bel):
        A = self.get_trans_mat_of(bel.t)
        Q = self.get_system_noise_of(bel.t)

        Sigma_pred = A @ bel.cov @ A.T + Q
        mean_pred = A @ bel.mean

        bel = bel.replace(
            mean=mean_pred,
            cov=Sigma_pred
        )
        return bel

    @partial(jax.jit, static_argnums=(0,))
    def update_state(self, bel, x, y):
        C = self.get_obs_mat_of(bel.t)
        R = self.get_observation_noise_of(bel.t)
        S = C @ bel.cov @ C.T + R
        K = jnp.linalg.solve(S, C @ bel.cov).T
        
        pred_obs = C @ bel.mean
        innovation = y - pred_obs
        mean = bel.mean + K @ innovation

        I = jnp.eye(len(mean))
        tmp = I - K @ C
        cov = tmp @ bel.cov @ tmp.T + K @ R @ K.T

        bel = bel.replace(
            mean=mean,
            cov=cov
        )
        return bel
    
    @partial(jax.jit, static_argnums=(0,))
    def predict_obs(self, bel, x):
        A = self.get_trans_mat_of(bel.t)
        y_pred = jnp.einsum("ij,...j->...i", A, x)
        return y_pred

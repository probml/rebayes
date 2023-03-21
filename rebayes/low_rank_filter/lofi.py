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

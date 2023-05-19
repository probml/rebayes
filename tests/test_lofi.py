import pytest

import jax.numpy as jnp
import jax.random as jr

from rebayes.low_rank_filter.lofi import (
    INFLATION_METHODS,
    RebayesLoFiOrthogonal,
    RebayesLoFiSpherical,
    RebayesLoFiDiagonal,
)
from rebayes.low_rank_filter.lofi_core import _fast_svd
from rebayes.datasets import rotating_permuted_mnist_data
from rebayes.utils.utils import get_mlp_flattened_params


RebayesLoFiEstimators = [RebayesLoFiOrthogonal, RebayesLoFiSpherical, RebayesLoFiDiagonal]


def setup_lofi(memory_size, steady_state, inflation_type, estimator_class):
    input_dim, hidden_dims, output_dim = 784, [2, 2], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    _, flat_params, _, apply_fn = get_mlp_flattened_params(model_dims)
    initial_mean, initial_covariance = flat_params, 1e-1
    estimator = estimator_class(
        dynamics_weights=1.0,
        dynamics_covariance=1e-1,
        emission_mean_function=apply_fn,
        emission_cov_function=None,
        adaptive_emission_cov=True,
        dynamics_covariance_inflation_factor=1e-5,
        memory_size=memory_size,
        steady_state=steady_state,
        inflation=inflation_type,
    )
    
    return initial_mean, initial_covariance, estimator
    

def test_fast_svd():
    for i in [10, 100, 1_000, 1_000_000]:
        print(i)
        A = jr.normal(jr.PRNGKey(i), (i, 10))
        u_n, s_n, _ = jnp.linalg.svd(A, full_matrices=False)
        u_s, s_s = _fast_svd(A)
        
        assert jnp.allclose(jnp.abs(u_n), jnp.abs(u_s), atol=1e-2) and jnp.allclose(s_n, s_s, atol=1e-2)
        

@pytest.mark.parametrize(
    "memory_size, steady_state, inflation_type, estimator_class",
    [(10, ss, it, ec) for ss in [True, False] for it in INFLATION_METHODS for ec in RebayesLoFiEstimators]
)
def test_lofi(memory_size, steady_state, inflation_type, estimator_class):
    # Load rotated MNIST dataset
    n_train = 200
    X_train, y_train = rotating_permuted_mnist_data.generate_rotating_mnist_dataset()
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    
    # Define mean callback function
    def callback(bel, *args, **kwargs):
        return bel.mean
    
    # Test if run without error
    initial_mean, initial_cov, lofi_estimator = \
        setup_lofi(memory_size, steady_state, inflation_type, estimator_class)
        
    _ = lofi_estimator.scan(initial_mean, initial_cov, X_train, y_train, callback)
    
    assert True
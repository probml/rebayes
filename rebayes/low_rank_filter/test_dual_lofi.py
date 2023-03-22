import pytest

import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from rebayes.dual_base import dual_rebayes_scan, DualRebayesParams, ObsModel
from rebayes.low_rank_filter.lofi import (
    LoFiParams,
    INFLATION_METHODS,
    RebayesLoFiOrthogonal,
    RebayesLoFiSpherical,
    RebayesLoFiDiagonal,
)
from rebayes.low_rank_filter.dual_lofi import (
    DualLoFiParams,
    make_dual_lofi_orthogonal_estimator,
    make_dual_lofi_spherical_estimator,
    make_dual_lofi_diagonal_estimator,
)
from rebayes.extended_kalman_filter.test_ekf import make_linreg_rebayes_params, make_linreg_data, make_linreg_prior
from rebayes.utils.utils import get_mlp_flattened_params


RebayesLoFiEstimators = [RebayesLoFiOrthogonal, RebayesLoFiSpherical, RebayesLoFiDiagonal]
make_dual_lofi_estimator_methods = [make_dual_lofi_orthogonal_estimator, make_dual_lofi_spherical_estimator, make_dual_lofi_diagonal_estimator]


def allclose(u, v):
    return jnp.allclose(u, v, atol=1e-2)

def make_linreg_dual_params(nfeatures):
    (obs_var, mu0, Sigma0) = make_linreg_prior()
    
    # Define Linear Regression as MLP with no hidden layers
    input_dim, hidden_dims, output_dim = nfeatures, [], 1
    model_dims = [input_dim, *hidden_dims, output_dim]
    *_, apply_fn = get_mlp_flattened_params(model_dims)
    
    params = DualRebayesParams(
        mu0=mu0,
        eta0=1/Sigma0[0,0],
        dynamics_scale_factor = 1.0,
        dynamics_noise = 0.0,
        obs_noise = obs_var,
        cov_inflation_factor = 0,
    )
    obs_model = ObsModel(
        emission_mean_function = lambda w, x: apply_fn(w, x),
        emission_cov_function = lambda w, x: obs_var
    )

    return params, obs_model


@pytest.mark.parametrize(
    "steady_state, inflation_type, estimator_class, make_dual_estimator_method",
    [(ss, it, RebayesLoFiEstimators[i], make_dual_lofi_estimator_methods[i]) 
     for ss in [True, False] for it in INFLATION_METHODS for i in range(len(RebayesLoFiEstimators))]
)
def test_lofi_adaptive_backwards_compatibility(steady_state, inflation_type, estimator_class, make_dual_estimator_method):
    (X, Y) = make_linreg_data()

    # old estimator
    _, D = X.shape
    params = make_linreg_rebayes_params(D)
    params.adaptive_emission_cov = True
    lofi_params = LoFiParams(memory_size=1, steady_state=steady_state, inflation=inflation_type)
    estimator = estimator_class(params, lofi_params)
    final_bel, _ = estimator.scan(X, Y)
    obs_noise_lofi = final_bel.obs_noise_var

    params, obs_model = make_linreg_dual_params(D)
    params.obs_noise = 1.0
    dual_lofi_params = DualLoFiParams(
        memory_size=1,
        inflation=inflation_type,
        steady_state=steady_state,
        obs_noise_estimator = "post",
        obs_noise_lr_fn= lambda t: 1.0/(t+1)
    )

    dual_estimator = make_dual_estimator_method(params, obs_model, dual_lofi_params)
    carry, _ = dual_rebayes_scan(dual_estimator,  X, Y)
    params, final_bel = carry
    obs_noise_dual = params.obs_noise
    
    assert allclose(obs_noise_dual, obs_noise_lofi)
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from rebayes.dual_base import dual_rebayes_scan, DualRebayesParams, ObsModel
from rebayes.extended_kalman_filter.ekf import RebayesEKF
from rebayes.extended_kalman_filter.dual_ekf import make_dual_ekf_estimator, EKFParams
from rebayes.extended_kalman_filter.test_ekf import make_linreg_rebayes_params, run_kalman, make_linreg_data, make_linreg_prior
from rebayes.utils.utils import get_mlp_flattened_params


def allclose(u, v):
    return jnp.allclose(u, v, atol=1e-3)

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


def test_linreg():
    # check that dual estimator matches KF for linear regression
    (X, Y) = make_linreg_data()
    lgssm_posterior = run_kalman(X, Y)
    mu_kf = lgssm_posterior.filtered_means
    cov_kf = lgssm_posterior.filtered_covariances
    ll_kf = lgssm_posterior.marginal_loglik

    N,D = X.shape
    params, obs_model = make_linreg_dual_params(D)
    ekf_params = EKFParams(method="fcekf")
    estimator = make_dual_ekf_estimator(params, obs_model, ekf_params)

    def callback(params, bel, pred_obs, t, u, y, bel_pred):
        m = pred_obs # estimator.predict_obs(params, bel_pred, u)
        P = estimator.predict_obs_cov(params, bel_pred, u)
        ll = MVN(m, P).log_prob(jnp.atleast_1d(y))
        return ll

    carry, lls = dual_rebayes_scan(estimator,  X, Y, callback)
    params, final_bel = carry
    # print(carry)
    T = mu_kf.shape[0]
    assert allclose(final_bel.mean, mu_kf[T-1])
    assert allclose(final_bel.cov, cov_kf[T-1])
    ll = jnp.sum(lls)
    assert jnp.allclose(ll, ll_kf, atol=1e-1)


def test_adaptive_backwards_compatibility():
    # check that we estimate the same obs noise as Peter's EKF code (for certain settings)
    (X, Y) = make_linreg_data()

    # old estimator
    N, D = X.shape
    params  = make_linreg_rebayes_params(D)
    params.adaptive_emission_cov = True
    estimator = RebayesEKF(params, method='fcekf')
    final_bel, lls = estimator.scan(X, Y)
    obs_noise_ekf = jnp.atleast_1d(final_bel.obs_noise_var).ravel()
    # print(obs_noise_ekf)

    params, obs_model = make_linreg_dual_params(D)
    # if we use the post-update estimator, initialized with q=0 and lr=1/N(t), we should match peter's code
    params.obs_noise = 0.0 * jnp.eye(1)
    ekf_params = EKFParams(method="fcekf", obs_noise_estimator = "post", obs_noise_lr_fn= lambda t: 1.0/(t+1))

    estimator = make_dual_ekf_estimator(params, obs_model, ekf_params)
    carry, lls = dual_rebayes_scan(estimator,  X, Y)
    params, final_bel = carry
    obs_noise_dual = jnp.atleast_1d(params.obs_noise).ravel()
    # print(obs_noise_dual)
    assert jnp.allclose(obs_noise_dual, obs_noise_ekf)


def test_adaptive():
    (X, Y) = make_linreg_data()
    N, D = X.shape
    params, obs_model = make_linreg_dual_params(D)
    init_R =  0.1*jnp.std(Y) * jnp.eye(1)
    lr = 0.01

    params.obs_noise = init_R
    ekf_params = EKFParams(method="fcekf", obs_noise_estimator = "post", obs_noise_lr_fn=lambda t: lr)
    estimator = make_dual_ekf_estimator(params, obs_model, ekf_params)
    (params, final_bel), lls = dual_rebayes_scan(estimator,  X, Y)
    obs_noise_post = params.obs_noise

    params.obs_noise = init_R
    ekf_params = EKFParams(method="fcekf", obs_noise_estimator = "pre", obs_noise_lr_fn= lambda t: lr)
    estimator = make_dual_ekf_estimator(params, obs_model, ekf_params)
    (params,final_bel), lls = dual_rebayes_scan(estimator,  X, Y)
    obs_noise_pre = params.obs_noise

    # print("post ", obs_noise_post, "pre ", obs_noise_pre)
    assert True
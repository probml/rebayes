from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from rebayes import base
from jax.flatten_util import ravel_pytree
from jax import vmap, jit
from bayes_opt import BayesianOptimization
import optax
import tensorflow_probability.substrates.jax as tfp

from rebayes.extended_kalman_filter import ekf
from rebayes.low_rank_filter import lofi
from rebayes.low_rank_filter import cold_posterior_lofi
from rebayes.low_rank_filter import replay_lofi
from rebayes.sgd_filter import replay_sgd as rsgd

tfd = tfp.distributions


def bbf_lofi(
    log_init_cov,
    log_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    # Specify before running
    init_fn,
    train,
    test,
    callback,
    memory_size,
    inflation = "hybrid",
    lofi_method = "diagonal",
    callback_at_end=True,
    n_seeds=5,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    dynamics_weights = 1 - jnp.exp(log_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    if lofi_method == "diagonal":
        lofi_estimator = lofi.RebayesLoFiDiagonal
    elif lofi_method == "spherical":
        lofi_estimator = lofi.RebayesLoFiSpherical
    else:
        raise ValueError("method must be either 'diagonal' or 'spherical'")
    
    model_dict = init_fn(key=0)
    emission_mean_fn, emission_cov_fn, apply_fn = \
        model_dict["emission_mean_function"], model_dict["emission_cov_function"], model_dict["apply_fn"]
    
    estimator = lofi_estimator(
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_fn,
        emission_cov_function=emission_cov_fn,
        dynamics_covariance_inflation_factor=alpha,
        memory_size=memory_size,
        inflation=inflation,
        emission_dist=lambda mean, cov: tfd.Categorical(probs=mean),
    )
    
    test_callback_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                            "apply_fn": apply_fn, "key": jr.PRNGKey(0), **kwargs}

    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, X_train, y_train, progress_bar=False)
            metric = callback(bel, **test_callback_kwargs)
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, X_train, y_train, 
                                       progress_bar=False, callback=callback, **test_callback_kwargs)
            metric = metric.mean()
        result.append(metric)
    result = jnp.array(result).mean()
    
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8
        
    return result


def bbf_replay_lofi(
    log_init_cov,
    log_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    # Specify before running
    emission_mean_fn,
    emission_cov_fn,
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    buffer_size,
    dim_input,
    dim_output,
    memory_size,
    inflation = "hybrid",
    callback_at_end=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    dynamics_weights = 1 - jnp.exp(log_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    params = replay_lofi.ReplayLoFiParams(
        buffer_size=buffer_size,
        dim_input=dim_input,
        dim_output=dim_output,
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_fn,
        emission_cov_function=emission_cov_fn,
        dynamics_covariance_inflation_factor=alpha,
        memory_size=memory_size,
        inflation=inflation,
    )

    estimator = replay_lofi.RebayesReplayLoFiDiagonal(params)

    test_callback_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                            "apply_fn": apply_fn, "key": jr.PRNGKey(0), **kwargs}
    
    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric.mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e8
        
    return metric


def bbf_ekf(
    log_init_cov,
    log_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    # Specify before running
    init_fn,
    train,
    test,
    callback,
    method="fdekf",
    callback_at_end=True,
    n_seeds=5,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test

    dynamics_weights = 1 - jnp.exp(log_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    model_dict = init_fn(key=0)
    emission_mean_fn, emission_cov_fn, apply_fn = \
        model_dict["emission_mean_function"], model_dict["emission_cov_function"], model_dict["apply_fn"]
    
    estimator = ekf.RebayesEKF(
        dynamics_weights_or_function=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_fn,
        emission_cov_function=emission_cov_fn,
        dynamics_covariance_inflation_factor=alpha,
        emission_dist=lambda mean, cov: tfd.Categorical(probs=mean),
        method=method,
    )

    test_callback_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                            "apply_fn": apply_fn, "key": jr.PRNGKey(0), **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, X_train, y_train, progress_bar=False)
            metric = callback(bel, **test_callback_kwargs)
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, X_train, y_train, 
                                       progress_bar=False, callback=callback, **test_callback_kwargs)
            metric = metric.mean()
        result.append(metric)
    result = jnp.array(result).mean()
        
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8

    return result


def bbf_rsgd(
    log_learning_rate,
    # Specify before running
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    loss_fn,
    buffer_size,
    dim_output,
    callback_at_end=True,
    optimizer="sgd",
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, y_train = train
    X_test, y_test = test
    
    if optimizer == "sgd":
        opt = optax.sgd
    elif optimizer == "adam":
        opt = optax.adam
    else:
        raise ValueError("optimizer must be either 'sgd' or 'adam'")
    
    tx = opt(learning_rate=jnp.exp(log_learning_rate).item())
    
    @partial(jit, static_argnames=("applyfn",))
    def lossfn_fifo(params, counter, X, y, applyfn):
        logits = vmap(applyfn, (None, 0))(params, X).ravel()
        nll = loss_fn(logits, y.ravel())
        nll = nll.sum()
        loss = (nll * counter).sum() / counter.sum()
        return loss
    
    estimator = rsgd.FifoSGDLaplaceDiag(
        lossfn_fifo,
        apply_fn=apply_fn,
        init_params=flat_params,
        tx=tx,
        buffer_size=buffer_size,
        dim_features=[1, 28, 28, 1],
        dim_output=dim_output,
        n_inner=1,
    )
    
    test_callback_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                            "apply_fn": apply_fn, "key": jr.PRNGKey(0), **kwargs}
        
    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric.mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e6
        
    return metric


def create_optimizer(
    init_fn,
    bounds,
    train,
    test,
    callback=None,
    method="fdekf",
    random_state=0,
    verbose=2,
    callback_at_end=True,
    n_seeds=5,
    **kwargs
):
    """init_fn(key) is a function of random jax key"""
    X_train, _ = train
    _, *n_features = X_train.shape
    
    if "sgd" not in method:
        if "ekf" in method:
            bbf = bbf_ekf
        elif "replay_lofi" in method:
            bbf = bbf_replay_lofi
        elif "lofi" in method:
            bbf = bbf_lofi

        bbf_partial = partial(
            bbf,
            init_fn=init_fn,
            train=train,
            test=test,
            callback=callback,
            callback_at_end=callback_at_end,
            n_seeds=n_seeds,
            **kwargs
        )
        if "ekf" in method:
            bbf_partial = partial(bbf_partial, method=method)
            
    else:
        bbf_partial = partial(
            bbf_rsgd,
            init_fn=init_fn,
            train=train,
            test=test,
            callback=callback,
            callback_at_end=callback_at_end,
            **kwargs # Must include loss_fn, buffer_size, dim_output if method is rsgd
        )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
        verbose=verbose,
        allow_duplicate_points=True,
    )

    return optimizer


def get_best_params(optimizer, method):
    max_params = optimizer.max["params"].copy()

    if "sgd" not in method:
        initial_covariance = jnp.exp(max_params["log_init_cov"]).item()
        dynamics_weights = 1 - jnp.exp(max_params["log_dynamics_weights"]).item()
        dynamics_covariance = jnp.exp(max_params["log_dynamics_cov"]).item()
        alpha = jnp.exp(max_params["log_alpha"]).item()

        hparams = {
            "initial_covariance": initial_covariance,
            "dynamics_covariance": dynamics_covariance,
            "dynamics_covariance_inflation_factor": alpha,
        }
        if "lofi" in method:
            hparams["dynamics_weights"] = dynamics_weights
        else:
            hparams["dynamics_weights_or_function"] = dynamics_weights
    else:
        learning_rate = jnp.exp(max_params["log_learning_rate"]).item()
        
        hparams = {
            "learning_rate": learning_rate,
        }

    return hparams


def build_estimator(model_dict, hparams, method, **kwargs):
    """
    _ is a dummy parameter for compatibility with lofi 
    """
    init_mean, apply_fn, emission_mean_fn, emission_cov_fn = \
        model_dict["flat_params"], model_dict["apply_fn"], \
            model_dict["emission_mean_function"], model_dict["emission_cov_function"]
    hparams = hparams.copy()
    if "ekf" in method:
        init_covariance = hparams.pop("initial_covariance")
        estimator = ekf.RebayesEKF(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=lambda mean, cov: tfd.Categorical(probs=mean),
            method=method,
            **hparams,
        )
        estimator = (estimator, init_mean, init_covariance)
    elif "replay_lofi" in method:
        params = replay_lofi.ReplayLoFiParams(
            initial_mean=init_mean,
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            **hparams,
            **kwargs,
        )
        estimator = replay_lofi.RebayesReplayLoFiDiagonal(params)
    elif "lofi" in method:
        if "lofi_method" in kwargs:
            if kwargs["lofi_method"] == "diagonal":
                estimator = lofi.RebayesLoFiDiagonal
            elif kwargs["lofi_method"] == "spherical":
                estimator = lofi.RebayesLoFiSpherical
            else:
                raise ValueError("method must be either 'diagonal' or 'spherical'")
        else:
            estimator = lofi.RebayesLoFiDiagonal
        kwargs.pop("lofi_method")
        init_covariance = hparams.pop("initial_covariance")
        estimator = estimator(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=lambda mean, cov: tfd.Categorical(probs=mean),
            **hparams,
            **kwargs,
        )
        estimator = (estimator, init_mean, init_covariance)
    elif "sgd" in method:
        if "optimizer" in kwargs:
            if kwargs["optimizer"] == "sgd":
                opt = optax.sgd
            elif kwargs["optimizer"] == "adam":
                opt = optax.adam
            else:
                raise ValueError("optimizer must be either 'sgd' or 'adam'")
        else:
            opt = optax.sgd
        tx = opt(learning_rate=hparams["learning_rate"])
        
        @partial(jit, static_argnames=("applyfn",))
        def lossfn_fifo(params, counter, X, y, applyfn):
            logits = vmap(applyfn, (None, 0))(params, X)
            nll = kwargs["loss_fn"](logits, y)
            nll = nll.sum()
            loss = (nll * counter).sum() / counter.sum()
            return loss
        
        estimator = rsgd.FifoSGD(
            lossfn_fifo,
            apply_fn=apply_fn,
            init_params=init_mean,
            tx=tx,
            buffer_size=kwargs["buffer_size"],
            dim_features=[1, 28, 28, 1],
            dim_output=kwargs["dim_output"],
            n_inner=1,
        )
    else:
        raise ValueError("method must be either 'ekf', 'lofi' or 'sgd'")
        
    return estimator
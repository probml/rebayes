from functools import partial

import jax
import jax.numpy as jnp
from rebayes import base
from jax.flatten_util import ravel_pytree
from jax import vmap, jit
from bayes_opt import BayesianOptimization
import optax

from rebayes.extended_kalman_filter import ekf
from rebayes.low_rank_filter import lofi
from rebayes.sgd_filter import replay_sgd as rsgd


def apply(flat_params, x, model, unflatten_fn):
    return model.apply(unflatten_fn(flat_params), x)


def bbf_lofi(
    log_init_cov,
    log_dynamics_weights,
    log_dynamics_cov,
    log_emission_cov,
    log_alpha,
    # Specify before running
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    memory_size,
    inflation = "hybrid",
    lofi_method = "diagonal",
    warmup_steps = 0,
    cooldown_steps=0,
    emission_mean_fn=None,
    emission_cov_fn=None,
    callback_at_end=True,
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
    if emission_mean_fn is None:
        emission_mean_fn = apply_fn
    if emission_cov_fn is None:
        def emission_cov_fn(w, x): return jnp.exp(log_emission_cov)

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params = lofi.LoFiParams(
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
    
    if lofi_method == "diagonal":
        lofi_estimator = lofi.RebayesLoFiDiagonal
    elif lofi_method == "spherical":
        lofi_estimator = lofi.RebayesLoFiSpherical
    else:
        raise ValueError("method must be either 'diagonal' or 'spherical'")

    estimator = lofi_estimator(params)

    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric[warmup_steps:-cooldown_steps].mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e8
        
    return metric


def bbf_ekf(
    log_init_cov,
    log_dynamics_weights,
    log_dynamics_cov,
    log_emission_cov,
    log_alpha,
    # Specify before running
    train,
    test,
    flat_params,
    callback,
    apply_fn,
    method="fdekf",
    warmup_steps = 0,
    cooldown_steps=0,
    emission_mean_fn=None,
    emission_cov_fn=None,
    callback_at_end=True,
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
    if emission_mean_fn is None:
        emission_mean_fn = apply_fn
    if emission_cov_fn is None:
        def emission_cov_fn(w, x): return jnp.exp(log_emission_cov)

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    params = ekf.EKFParams(
        initial_mean=flat_params,
        initial_covariance=initial_covariance,
        dynamics_weights_or_function=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=emission_mean_fn,
        emission_cov_function=emission_cov_fn,
        dynamics_covariance_inflation_factor=alpha,
    )

    estimator = ekf.RebayesEKF(params, method=method)

    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric[warmup_steps:-cooldown_steps].mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e8

    return metric


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
    warmup_steps=0,
    cooldown_steps=0,
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

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    
    @partial(jit, static_argnames=("applyfn",))
    def lossfn_fifo(params, counter, X, y, applyfn):
        logits = vmap(applyfn, (None, 0))(params, X).ravel()
        nll = loss_fn(logits, y.ravel())
        nll = nll.sum()
        loss = (nll * counter).sum() / counter.sum()
        return loss
    
    estimator = rsgd.FifoSGD(
        lossfn_fifo,
        apply_fn=apply_fn,
        init_params=flat_params,
        tx=tx,
        buffer_size=buffer_size,
        dim_features=[1, 28, 28, 1],
        dim_output=dim_output,
        n_inner=1,
    )
    
    if callback_at_end:
        bel, _ = estimator.scan(X_train, y_train, progress_bar=False)
        metric = callback(bel, **test_callback_kwargs)
    else:
        _, metric = estimator.scan(X_train, y_train, progress_bar=False, callback=callback, **test_callback_kwargs)
        metric = metric[warmup_steps:-cooldown_steps].mean()
        
    if jnp.isnan(metric) or jnp.isinf(metric):
        metric = -1e6
        
    return metric


def create_optimizer(
    model,
    bounds,
    random_state,
    train,
    test,
    callback=None,
    method="fdekf",
    verbose=2,
    callback_at_end=True,
    emission_mean_fn=None,
    emission_cov_fn=None,
    warmup_steps=0,
    cooldown_steps=0,
    **kwargs
):
    key = jax.random.PRNGKey(random_state)
    X_train, _ = train
    _, *n_params = X_train.shape

    batch_init = jnp.ones((1, *n_params))
    params_init = model.init(key, batch_init)
    flat_params, recfn = ravel_pytree(params_init)
    
    apply_fn = partial(apply, model=model, unflatten_fn=recfn)

    if "sgd" not in method:
        if "ekf" in method:
            bbf = bbf_ekf
        elif "lofi" in method:
            bbf = bbf_lofi

        bbf_partial = partial(
            bbf,
            emission_mean_fn=emission_mean_fn,
            emission_cov_fn=emission_cov_fn,
            train=train,
            test=test,
            flat_params=flat_params,
            callback=callback,
            apply_fn=apply_fn,
            callback_at_end=callback_at_end,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            **kwargs
        )
        if "ekf" in method:
            bbf_partial = partial(bbf_partial, method=method)
        # Fix log-emission-covariance to dummy if adaptive
        if emission_cov_fn is not None:
            bbf_partial = partial(
                bbf_partial,
                log_emission_cov=0.0,
            )
    else:
        bbf_partial = partial(
            bbf_rsgd,
            train=train,
            test=test,
            flat_params=flat_params,
            callback=callback,
            apply_fn=apply_fn,
            callback_at_end=callback_at_end,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            **kwargs # Must include loss_fn, buffer_size, dim_output if method is rsgd
        )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
        allow_duplicate_points=True,
        verbose=verbose,
    )

    return optimizer, apply_fn, n_params


def get_best_params(optimizer, method):
    if type(optimizer) is dict:
        max_params = optimizer.copy()
    else:
        max_params = optimizer.max["params"].copy()

    if "sgd" not in method:
        initial_covariance = jnp.exp(max_params["log_init_cov"]).item()
        dynamics_weights = 1 - jnp.exp(max_params["log_dynamics_weights"]).item()
        dynamics_covariance = jnp.exp(max_params["log_dynamics_cov"]).item()
        emission_cov = jnp.exp(max_params.get("log_emission_cov", 0.0)).item()
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
        hparams["emission_cov_function"] = lambda w, x: emission_cov
        
    else:
        learning_rate = jnp.exp(max_params["log_learning_rate"]).item()
        
        hparams = {
            "learning_rate": learning_rate,
        }

    return hparams


def build_estimator(init_mean, apply_fn, hparams, method,
                    emission_mean_fn=None, emission_cov_function=None, **kwargs):
    if emission_mean_fn is None:
        emission_mean_fn = apply_fn
    if "ekf" in method:
        params = ekf.EKFParams(
            initial_mean=init_mean,
            emission_mean_function=emission_mean_fn,
            **hparams,
        )
        estimator = ekf.RebayesEKF(params, method=method)
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
        params = lofi.LoFiParams(
            initial_mean=init_mean,
            emission_mean_function=emission_mean_fn,
            **hparams,
            **kwargs,
        )
        estimator = estimator(params)
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
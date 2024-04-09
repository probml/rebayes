from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
from bayes_opt import BayesianOptimization
import optax
import tensorflow_probability.substrates.jax as tfp

from rebayes.extended_kalman_filter import ekf
from rebayes.extended_kalman_filter import enkf
from rebayes.extended_kalman_filter import replay_ekf
from rebayes.ivon import ivon
from rebayes.low_rank_filter import lofi
from rebayes.sgd_filter import replay_sgd as rsgd
import rebayes.utils.normalizing_flows as nf

tfd = tfp.distributions


def bbf_lofi(
    log_init_cov,
    log_1m_dynamics_weights,
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
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_weights = 1 - jnp.exp(log_1m_dynamics_weights).item()
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
    
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    
    estimator = lofi_estimator(
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        dynamics_covariance_inflation_factor=alpha,
        memory_size=memory_size,
        inflation=inflation,
        emission_dist=emission_dist,
    )
    
    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test,
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0),
                      **kwargs}

    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
    
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8
        
    return result


def bbf_lofi_it(
    log_init_cov,
    log_1m_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    log_learning_rate,
    # Specify before running
    n_replay,
    init_fn,
    train,
    test,
    callback,
    memory_size,
    inflation = "hybrid",
    lofi_method = "diagonal",
    callback_at_end=True,
    n_seeds=5,
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_weights = 1 - jnp.exp(log_1m_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()
    learning_rate = jnp.exp(log_learning_rate).item()

    if lofi_method == "diagonal":
        lofi_estimator = lofi.RebayesIteratedLoFi
    else:
        raise ValueError("method must be 'diagonal'")
    
    model_dict = init_fn(key=0)
    
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    
    estimator = lofi_estimator(
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        dynamics_covariance_inflation_factor=alpha,
        memory_size=memory_size,
        inflation=inflation,
        emission_dist=emission_dist,
        n_replay=n_replay,
        learning_rate=learning_rate,
    )
    
    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test,
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0),
                      **kwargs}

    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
    
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8
        
    return result


def bbf_lofi_grad(
    log_init_cov,
    log_1m_dynamics_weights,
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
    correction_method="momentum-correction",
    momentum_weight=0.0,
    gradient_avg=True,
    callback_at_end=True,
    n_seeds=5,
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_weights = 1 - jnp.exp(log_1m_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    if lofi_method == "diagonal":
        lofi_estimator = lofi.RebayesGradientLoFi
    else:
        raise ValueError("method must be 'diagonal'")
    
    model_dict = init_fn(key=0)
    
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    estimator = lofi_estimator(
        dynamics_weights=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        dynamics_covariance_inflation_factor=alpha,
        memory_size=memory_size,
        inflation=inflation,
        emission_dist=emission_dist,
        correction_method=correction_method,
        momentum_weight=momentum_weight,
        gradient_avg=gradient_avg,
    )
    
    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test,
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0),
                      **kwargs}

    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
    
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8
        
    return result


def bbf_ekf(
    log_init_cov,
    log_1m_dynamics_weights,
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
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_weights = 1 - jnp.exp(log_1m_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    model_dict = init_fn(key=0)
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    estimator = ekf.RebayesEKF(
        dynamics_weights_or_function=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        dynamics_covariance_inflation_factor=alpha,
        emission_dist=emission_dist,
        method=method,
    )

    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0), 
                      **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
        
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8

    return result


def bbf_ekf_it(
    log_init_cov,
    log_1m_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    log_learning_rate,
    # Specify before running
    n_replay,
    init_fn,
    train,
    test,
    callback,
    method="fdekf-it",
    callback_at_end=True,
    n_seeds=5,
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_weights = 1 - jnp.exp(log_1m_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()
    learning_rate = jnp.exp(log_learning_rate).item()

    method_name = method.split("-")[0]
    model_dict = init_fn(key=0)
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    estimator = replay_ekf.RebayesReplayEKF(
        dynamics_weights_or_function=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        dynamics_covariance_inflation_factor=alpha,
        emission_dist=emission_dist,
        n_replay=n_replay,
        learning_rate=learning_rate,
        method=method_name,
    )

    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0), 
                      **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
        
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8

    return result


def bbf_ekf_ocl(
    log_init_cov,
    log_1m_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    log_learning_rate,
    # Specify before running
    init_fn,
    train,
    test,
    callback,
    method="fdekf-ocl",
    callback_at_end=True,
    n_seeds=5,
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_decay_delta = -2*jnp.log(1 - jnp.exp(log_1m_dynamics_weights)).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()
    learning_rate = jnp.exp(log_learning_rate).item()

    method_name = method.split("-")[0]
    model_dict = init_fn(key=0)
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    estimator = ekf.RebayesOCLEKF(
        dynamics_decay_delta=dynamics_decay_delta,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        dynamics_covariance_inflation_factor=alpha,
        emission_dist=emission_dist,
        learning_rate=learning_rate,
        method=method_name,
        decay_dynamics_weight=False,
    )

    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0), 
                      **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
        
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8

    return result


def bbf_ekf_nf(
    log_init_cov,
    log_1m_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    log_learning_rate,
    # Specify before running
    dim_input,
    dim_output,
    buffer_size,
    init_fn,
    train,
    test,
    callback,
    method="fdekf-nf",
    callback_at_end=True,
    n_seeds=5,
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_weights = 1 - jnp.exp(log_1m_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()
    learning_rate = jnp.exp(log_learning_rate).item()

    method_name = method.split("-")[0]
    model_dict = init_fn(key=0)
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
        
    nf_model = nf.NF_MLP()
    flat_params = model_dict["flat_params"]
    input_dim = len(flat_params)
    nfs = nf.init_normalizing_flow(nf_model, input_dim)
    
    estimator = ekf.RebayesNFEKF(
        buffer_size=buffer_size,
        dim_input=dim_input,
        dim_output=dim_output,
        dynamics_weights_or_function=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        nf_initial_params=nfs["params"],
        nf_apply_function=nfs["apply_fn"],
        dynamics_covariance_inflation_factor=alpha,
        emission_dist=emission_dist,
        learning_rate=learning_rate,
        method=method_name,
    )

    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0), 
                      **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
        
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8

    return result


def bbf_enkf(
    log_init_cov,
    log_1m_dynamics_weights,
    log_dynamics_cov,
    log_alpha,
    # Specify before running
    init_fn,
    train,
    test,
    callback,
    n_particles=10,
    callback_at_end=True,
    n_seeds=5,
    classification=True,
    **kwargs,
):
    """
    Black-box function for Bayesian optimization.
    """
    X_train, *_, y_train = train
    X_test, *_, y_test = test

    dynamics_weights = 1 - jnp.exp(log_1m_dynamics_weights).item()
    dynamics_covariance = jnp.exp(log_dynamics_cov).item()
    initial_covariance = jnp.exp(log_init_cov).item()
    alpha = jnp.exp(log_alpha).item()

    model_dict = init_fn(key=0)
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    estimator = enkf.RebayesEnKF(
        dynamics_weights_or_function=dynamics_weights,
        dynamics_covariance=dynamics_covariance,
        emission_mean_function=model_dict["emission_mean_function"],
        emission_cov_function=model_dict["emission_cov_function"],
        dynamics_covariance_inflation_factor=alpha,
        emission_dist=emission_dist,
        n_particles=n_particles,
    )

    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0), 
                      **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, 
                                    X_train, y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, 
                                       X_train, y_train, progress_bar=False, 
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
        
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e8

    return result


def bbf_rsgd(
    log_init_cov,
    log_learning_rate,
    log_1m_momentum,
    # Specify before running
    init_fn,
    train,
    test,
    callback,
    loss_fn,
    buffer_size,
    dim_input,
    dim_output,
    callback_at_end=True,
    optimizer="sgd",
    n_seeds=5,
    **kwargs,
):
    X_train, *_, y_train = train
    X_test, *_, y_test = test
    
    initial_covariance = jnp.exp(log_init_cov).item()
    learning_rate=jnp.exp(log_learning_rate).item()
    momentum = 1 - jnp.exp(log_1m_momentum).item()
    if optimizer == "sgd":
        tx = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        tx = optax.adam(learning_rate=learning_rate)
    else:
        raise ValueError("optimizer must be either 'sgd' or 'adam'")
    
    model_dict = init_fn(key=0)
    
    @partial(jit, static_argnames=("applyfn",))
    def lossfn_fifo(params, counter, X, y, applyfn):
        logits = vmap(applyfn, (None, 0))(params, X).ravel()
        nll = loss_fn(logits, y.ravel())
        nll = nll.sum()
        loss = (nll * counter).sum() / counter.sum()
        return loss
    
    @partial(jit, static_argnames=("applyfn",))
    def loglikelihood_fifo(params, X, y, applyfn):
        logits = vmap(applyfn, (None, 0))(params, X).ravel()
        ll = -loss_fn(logits, y.ravel())
        ll = ll.sum()
        return ll
    
    estimator = rsgd.FifoSGDLaplaceDiag(
        lossfn_fifo,
        loglikelihood_fifo,
        apply_fn=model_dict["apply_fn"],
        emission_cov_function=model_dict["emission_cov_function"],
        tx=tx,
        buffer_size=buffer_size,
        dim_features=dim_input,
        dim_output=dim_output,
        n_inner=1,
    )
    
    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0), 
                      **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, initial_covariance, X_train, 
                                    y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, initial_covariance, X_train, 
                                       y_train, progress_bar=False,
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
    
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e6
        
    return result


def bbf_ivon(
    log_learning_rate,
    # Specify before running
    init_fn,
    train,
    test,
    callback,
    loss_fn,
    n_sample,
    callback_at_end=True,
    n_seeds=5,
    **kwargs,
):
    X_train, *_, y_train = train
    X_test, *_, y_test = test
    
    learning_rate=jnp.exp(log_learning_rate).item()
    model_dict = init_fn(key=0)
    
    estimator = ivon.RebayesIVON(
        apply_fn=model_dict["apply_fn"],
        loss_fn=loss_fn,
        n_sample=n_sample,
        beta_1=0.9,
        beta_2=0.99999,
        learning_rate=learning_rate,
        weight_decay=1e-4,
    )
    
    test_cb_kwargs = {"agent": estimator, "X_test": X_test, "y_test": y_test, 
                      "apply_fn": model_dict["apply_fn"], "key": jr.PRNGKey(0), 
                      **kwargs}
    
    result = []
    for i in range(n_seeds):
        model_dict = init_fn(key=i)
        flat_params = model_dict["flat_params"]
        if callback_at_end:
            bel, _ = estimator.scan(flat_params, 0.01, X_train, 
                                    y_train, progress_bar=False)
            metric = jnp.array(list(callback(bel, **test_cb_kwargs).values()))
        else:
            _, metric = estimator.scan(flat_params, 0.01, X_train, 
                                       y_train, progress_bar=False,
                                       callback=callback, **test_cb_kwargs)
            metric = jnp.array(list(metric.values())).mean()
        result.append(metric)
    result = jnp.array(result).mean()
    
    if jnp.isnan(result) or jnp.isinf(result):
        result = -1e6
        
    return result


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
    nll_method="nll",
    classification=True,
    **kwargs
):
    """init_fn(key) is a function of random jax key"""
    if "sgd" in method or "adam" in method:
        bbf_partial = partial(
            bbf_rsgd,
            init_fn=init_fn,
            train=train,
            test=test,
            callback=callback,
            callback_at_end=callback_at_end,
            n_seeds=n_seeds,
            **kwargs # Must include loss_fn, buffer_size, dim_output
        )
        if "log_1m_momentum" not in bounds:
            bbf_partial = partial(
                bbf_partial,
                log_1m_momentum=0.0
            )
        if nll_method == "nll":
            bbf_partial = partial(
                bbf_partial,
                log_init_cov=0.0
            )
    elif "ivon" in method:
        bbf_partial = partial(
            bbf_ivon,
            init_fn=init_fn,
            train=train,
            test=test,
            callback=callback,
            callback_at_end=callback_at_end,
            n_seeds=n_seeds,
            **kwargs
        )
    else:
        curr_method = method
        if "ekf-it" in method:
            bbf = bbf_ekf_it
        elif "ekf-ocl" in method:
            bbf = bbf_ekf_ocl
        elif "ekf-nf" in method:
            bbf = bbf_ekf_nf
        elif "ekf" in method:
            bbf = bbf_ekf
        elif "linear" in method:
            bbf = bbf_ekf
            curr_method = "fcekf"
        elif "enkf" in method:
            bbf = bbf_enkf
        elif "lofi-it" in method:
            bbf = bbf_lofi_it
        elif "lofi" in method and "grad" in method:
            bbf = bbf_lofi_grad
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
            method=curr_method,
            classification=classification,
            **kwargs
        )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds,
        random_state=random_state,
        verbose=verbose,
        allow_duplicate_points=True,
    )

    return optimizer


def get_best_params(optimizer, method, nll_method="nll"):
    max_params = optimizer.max["params"].copy()
    if "sgd" in method or "adam" in method:
        learning_rate = jnp.exp(max_params["log_learning_rate"]).item()
        hparams = {
            "learning_rate": learning_rate,
        }
        if "log_1m_momentum" in max_params:
            momentum = 1 - jnp.exp(max_params["log_1m_momentum"]).item()
            hparams["momentum"] = momentum
        if nll_method != "nll":
            initial_covariance = jnp.exp(max_params["log_init_cov"]).item()
            hparams["initial_covariance"] = initial_covariance
    elif "ivon" in method:
        learning_rate = jnp.exp(max_params["log_learning_rate"]).item()
        
        hparams = {
            "initial_hessian": 0.01,
            "learning_rate": learning_rate,
        }
    else:
        initial_covariance = jnp.exp(max_params["log_init_cov"]).item()
        dynamics_weights = \
            1 - jnp.exp(max_params["log_1m_dynamics_weights"]).item()
        dynamics_covariance = jnp.exp(max_params["log_dynamics_cov"]).item()
        alpha = jnp.exp(max_params["log_alpha"]).item()
        
        hparams = {
            "initial_covariance": initial_covariance,
            "dynamics_covariance": dynamics_covariance,
            "dynamics_covariance_inflation_factor": alpha,
        }
        if "lofi" in method:
            hparams["dynamics_weights"] = dynamics_weights
        elif "ocl" in method:
            hparams["dynamics_decay_delta"] = -2*jnp.log(1 - dynamics_weights).item()            
        else:
            hparams["dynamics_weights_or_function"] = dynamics_weights
            
        if "it" in method or "ocl" in method or "nf" in method:
            hparams["learning_rate"] = jnp.exp(max_params["log_learning_rate"]).item()

    return hparams


def build_estimator(init_fn, hparams, method, classification=True, **kwargs):
    model_dict = init_fn(key=0)
    apply_fn, emission_mean_fn, emission_cov_fn = \
        model_dict["apply_fn"], model_dict["emission_mean_function"], \
            model_dict["emission_cov_function"]
    hparams = hparams.copy()
    emission_dist = lambda mean, cov: tfd.OneHotCategorical(probs=mean) \
        if classification else tfd.Normal(loc=mean, scale=jnp.sqrt(cov))
    if "ekf-it" in method:
        init_covariance = hparams.pop("initial_covariance")
        method_name = method.split("-")[0]
        estimator = replay_ekf.RebayesReplayEKF(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=emission_dist,
            method=method_name,
            **hparams,
            **kwargs,
        )
    elif "ekf-ocl" in method:
        init_covariance = hparams.pop("initial_covariance")
        method_name = method.split("-")[0]
        estimator = ekf.RebayesOCLEKF(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=emission_dist,
            method=method_name,
            decay_dynamics_weight=False,
            **hparams,
            **kwargs,
        )
    elif "ekf-nf" in method:
        init_covariance = hparams.pop("initial_covariance")
        method_name = method.split("-")[0]
        nf_model = nf.NF_MLP()
        flat_params = model_dict["flat_params"]
        input_dim = len(flat_params)
        nfs = nf.init_normalizing_flow(nf_model, input_dim)
        
        estimator = ekf.RebayesNFEKF(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=emission_dist,
            nf_initial_params=nfs["params"],
            nf_apply_function=nfs["apply_fn"],
            method=method_name,
            buffer_size=kwargs["buffer_size"],
            dim_input=kwargs["dim_input"],
            dim_output=kwargs["dim_output"],
            **hparams,
            **kwargs,
        )
    elif "ekf" in method or "linear" in method:
        if "linear" in method:
            method = "fcekf"
        init_covariance = hparams.pop("initial_covariance")
        estimator = ekf.RebayesEKF(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=emission_dist,
            method=method,
            **hparams,
        )
    elif "enkf" in method:
        init_covariance = hparams.pop("initial_covariance")
        estimator = enkf.RebayesEnKF(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=emission_dist,
            **hparams,
            **kwargs,
        )
    elif "lofi" in method:
        if "lofi_method" in kwargs:
            if kwargs["lofi_method"] == "diagonal":
                estimator = lofi.RebayesLoFiDiagonal
            elif kwargs["lofi_method"] == "spherical":
                estimator = lofi.RebayesLoFiSpherical
            else:
                raise ValueError("method must be either 'diagonal' or 'spherical'")
            kwargs.pop("lofi_method")
        else:
            estimator = lofi.RebayesLoFiDiagonal
        if "it" in method:
            estimator = lofi.RebayesIteratedLoFi
        if "grad" in method:
            estimator = lofi.RebayesGradientLoFi
        init_covariance = hparams.pop("initial_covariance")
        estimator = estimator(
            emission_mean_function=emission_mean_fn,
            emission_cov_function=emission_cov_fn,
            emission_dist=emission_dist,
            **hparams,
            **kwargs,
        )
    elif "sgd" in method or "adam" in method:
        if "optimizer" in kwargs:
            if kwargs["optimizer"] == "sgd":
                if "momentum" in hparams:
                    tx = optax.sgd(
                        learning_rate=hparams["learning_rate"],
                        momentum=hparams["momentum"],
                    )
                else:
                    tx = optax.sgd(
                        learning_rate=hparams["learning_rate"],
                    )
            elif kwargs["optimizer"] == "adam":
                tx = optax.adam(
                    learning_rate=hparams["learning_rate"],
                )
            else:
                raise ValueError("optimizer must be either 'sgd' or 'adam'")
        else:
            tx = optax.sgd(learning_rate=hparams["learning_rate"])
        
        @partial(jit, static_argnames=("applyfn",))
        def lossfn_fifo(params, counter, X, y, applyfn):
            logits = vmap(applyfn, (None, 0))(params, X)
            nll = kwargs["loss_fn"](logits, y)
            nll = nll.sum()
            loss = (nll * counter).sum() / counter.sum()
            return loss
        
        @partial(jit, static_argnames=("applyfn",))
        def loglikelihood_fifo(params, X, y, applyfn):
            logits = vmap(applyfn, (None, 0))(params, X).ravel()
            ll = -kwargs["loss_fn"](logits, y.ravel())
            ll = ll.sum()
            return ll
        
        estimator = rsgd.FifoSGDLaplaceDiag(
            lossfn_fifo,
            loglikelihood_fifo,
            apply_fn=apply_fn,
            emission_cov_function=emission_cov_fn,
            tx=tx,
            buffer_size=kwargs["buffer_size"],
            dim_features=kwargs["dim_input"],
            dim_output=kwargs["dim_output"],
            n_inner=1,
        )
        init_covariance = 1.0
        if "initial_covariance" in hparams:
            init_covariance = hparams.pop("initial_covariance")
    elif "ivon" in method:
        init_covariance = hparams.pop("initial_hessian") # Relabeling for consistency!
        estimator = ivon.RebayesIVON(
            apply_fn=apply_fn,
            loss_fn=kwargs["loss_fn"],
            n_sample=kwargs["n_sample"],
            **hparams,
        )
    else:
        raise ValueError("method must be either 'ekf', 'lofi', 'sgd' or 'ivon'")

    result = {
        "agent": estimator,
        "init_cov": init_covariance,
    }
        
    return result
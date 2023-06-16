"""
Custom callbacks
"""

from functools import partial

import jax
import jax.numpy as jnp
import distrax
import optax


# ------------------------------------------------------------------------------
# Common Callbacks

@partial(jax.jit, static_argnums=(1,4,5))
def evaluate_function(flat_params, apply_fn, X_test, y_test, loss_fn, 
                      label="loss", **kwargs):
    def evaluate(label, image):
        image = image.reshape((1, 28, 28, 1))
        logits = apply_fn(flat_params, image).ravel()
        loss = loss_fn(logits, label, **kwargs)
        
        return loss
    evals = jax.vmap(evaluate, (0, 0))(y_test, X_test)
    result = {
        label: evals.mean()
    }
    
    return result
    

def cb_eval(bel, *args, evaluate_fn, nan_val=-1e8, **kwargs):
    X, y, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    eval = evaluate_fn(bel.mean, apply_fn, X, y)
    if isinstance(eval, dict):
        eval = {k: jnp.where(jnp.isnan(v), nan_val, v) for k, v in eval.items()}
    else:
        eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    
    return eval


def cb_osa(bel, y_pred, t, X, y, bel_pred, evaluate_fn, nan_val=-1e8, 
           label="loss", **kwargs):
    eval = evaluate_fn(y_pred, y)
    eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    result = {
        label: eval.mean()
    }
    
    return result


# ------------------------------------------------------------------------------
# Regression

def cb_clf_sup(bel, pred_obs, t, X, y, bel_pred, apply_fn, lagn=20, store_fro=True, **kwargs):
    """
    Callback for a classification task with a supervised loss function.
    """
    X_test, y_test = kwargs["X_test"], kwargs["y_test"]
    recfn = kwargs["recfn"]

    slice_ix = jnp.arange(0, lagn) + t

    X_test = jnp.take(X_test, slice_ix, axis=0, fill_value=0)
    y_test = jnp.take(y_test, slice_ix, axis=0, fill_value=0)

    y_next = y.squeeze().argmax()
    phat_next = pred_obs.squeeze()
    yhat_next = phat_next.argmax()

    yhat_test = apply_fn(bel.mean, X_test).squeeze().argmax()

    # Compute errors
    err_test = (y_test == yhat_test).mean()
    err = (y_next == yhat_next).mean()

    if store_fro:
        mean_params = recfn(bel.mean)
        params_magnitude = jax.tree_map(lambda A: A["kernel"], mean_params, is_leaf=lambda k: "kernel" in k)
        params_magnitude = jax.tree_map(lambda A: jnp.linalg.norm(A, ord="fro"), params_magnitude)
    else:
        params_magnitude = None

    res = {
        "n-step-pred": yhat_test,
        "nsa-error": err_test,
        "osa-error": err,
        "phat": phat_next,
        "params_magnitude": params_magnitude
    }
    return res


def cb_reg_sup(bel, pred_obs, t, X, y, bel_pred, apply_fn, ymean, ystd, steps=10, **kwargs):
    """
    Callback for a regression task with a supervised loss function.
    """
    X_test, y_test = kwargs["X_test"], kwargs["y_test"]

    slice_ix = jnp.arange(0, steps) + t - steps // 2

    # eval on all tasks test set
    yhat_test = jax.vmap(apply_fn, (None, 0))(bel_pred.mean, X_test).squeeze()

    # De-normalise target variables
    y_test = y_test * ystd + ymean
    yhat_test = yhat_test.ravel() * ystd + ymean

    y_next = y.ravel() * ystd + ymean
    yhat_next = pred_obs.ravel() * ystd + ymean

    # Compute errors
    err_test = jnp.power(y_test - yhat_test, 2)
    err_test_window = err_test[slice_ix].mean()
    err_test = err_test.mean()
    err = jnp.power(y_next - yhat_next, 2).mean()

    err = jnp.sqrt(err)
    err_test = jnp.sqrt(err_test)
    err_test_window = jnp.sqrt(err_test_window)

    res = {
        "n-step-pred": yhat_test,
        "osa-metric": err, # one-step ahead
        "test-metric": err_test, # full dataset
        "window-metric": err_test_window, # window
    }

    return res


def cb_reg_mc(bel, pred_obs, t, X, y, bel_pred, apply_fn, steps=10, **kwargs):
    agent = kwargs["agent"]
    scale = kwargs["scale"]
    X_test, y_test = kwargs["X_test"], kwargs["y_test"]
    key = jax.random.fold_in(kwargs["key"], t)
    slice_ix = jnp.arange(0, steps) + t - steps // 2
    mean_test = apply_fn(bel_pred.mean, X_test).squeeze()
    nll = -distrax.Normal(pred_obs, scale).log_prob(y.ravel())
    nll_test = -distrax.Normal(mean_test, scale).log_prob(y_test.ravel())
    nll_window = nll_test[slice_ix].mean()
    nll_test = nll_test.mean()

    nlpd = agent.nlpd_mc(bel_pred, key, X, y).mean()
    nlpd_test = agent.nlpd_mc(bel_pred, key, X_test, y_test[:, None]).mean()
    nlpd_window = agent.nlpd_mc(bel_pred, key, X_test[slice_ix], y_test[slice_ix][:, None]).mean()

    res = cb_reg_sup(
        bel, pred_obs, t, X, y, bel_pred, apply_fn, **kwargs
    )

    res = {
        **res,
        "nlpd": nlpd,
        "nlpd_test": nlpd_test,
        "nlpd_window": nlpd_window,
        "nll": nll,
        "nll_test": nll_test,
        "nll_window": nll_window,
    }

    return res
    

# Minimal version for hparam tuning
def cb_reg_nlpd_mc(bel, pred_obs, t, X, y, bel_pred, nan_val=-1e8, **kwargs):
    X_test, y_test, apply_fn, agent = \
        kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"], kwargs["agent"]
    key = jax.random.fold_in(kwargs["key"], t)
    nlpd = agent.nlpd_mc(bel, key, X_test[:, jnp.newaxis, :], y_test).mean()
    nlpd = {
        "nlpd": jnp.where(jnp.isnan(nlpd), nan_val, nlpd)
    }
    
    return nlpd


# Evaluation functions
ll_reg = lambda pred_obs, y, scale: distrax.Normal(pred_obs, scale).log_prob(y).mean()
nll_reg = lambda pred_obs, y, scale: -ll_reg(pred_obs, y, scale)
generate_ll_reg_eval_fn = lambda scale: \
    partial(evaluate_function, label="ll", loss_fn=partial(ll_reg, scale=scale))
generate_nll_reg_eval_fn = lambda scale: \
    partial(evaluate_function, label="nll", loss_fn=partial(nll_reg, scale=scale))
rmse_reg = lambda pred_obs, y: jnp.sqrt(jnp.mean(jnp.power(pred_obs - y, 2)))
nrmse_reg = lambda pred_obs, y: -rmse_reg(pred_obs, y)
rmse_reg_eval_fn = partial(evaluate_function, label="rmse", loss_fn=rmse_reg)
nrmse_reg_eval_fn = partial(evaluate_function, label="nrmse", loss_fn=nrmse_reg)

def reg_eval_fn(flat_params, apply_fn, X_test, y_test, scale):
    nll = generate_nll_reg_eval_fn(scale)(flat_params, apply_fn, X_test, y_test)
    rmse = rmse_reg_eval_fn(flat_params, apply_fn, X_test, y_test)
    result = {**nll, **rmse}
    
    return result

# ------------------------------------------------------------------------------
# Classification


def cb_clf_discrete_tasks(bel, pred_obs, t, x, y, bel_pred, i,
                          nll_loss_fn=None, miscl_loss_fn=None, **kwargs):
    if nll_loss_fn is None:
        nll_loss_fn = lambda logits, label: \
            optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
    if miscl_loss_fn is None:
        miscl_loss_fn = lambda logits, label: jnp.mean(logits.argmax(axis=-1) != label)
    
    nll_evaluate_fn = partial(
        evaluate_function,
        loss_fn=nll_loss_fn,
    )
    miscl_evaluate_fn = partial(
        evaluate_function,
        loss_fn=miscl_loss_fn,
    )
    
    X_test, y_test, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    ntest_per_batch = kwargs["ntest_per_batch"]
    
    prev_test_batch, curr_test_batch = i*ntest_per_batch, (i+1)*ntest_per_batch
    curr_X_test, curr_y_test = \
        X_test[prev_test_batch:curr_test_batch], y_test[prev_test_batch:curr_test_batch]
    cum_X_test, cum_y_test = X_test[:curr_test_batch], y_test[:curr_test_batch]
    
    overall_nll_result = nll_evaluate_fn(bel.mean, apply_fn, cum_X_test, cum_y_test, label="overall")
    current_nll_result = nll_evaluate_fn(bel.mean, apply_fn, curr_X_test, curr_y_test, label="current")
    task1_nll_result = nll_evaluate_fn(bel.mean, apply_fn, X_test[:ntest_per_batch], y_test[:ntest_per_batch],
                                       label="task1")
        
    nll_result = {**overall_nll_result, **current_nll_result, **task1_nll_result,}
    
    overall_miscl_result = miscl_evaluate_fn(bel.mean, apply_fn, cum_X_test, cum_y_test, label="overall")
    current_miscl_result = miscl_evaluate_fn(bel.mean, apply_fn, curr_X_test, curr_y_test, label="current")
    task1_miscl_result = miscl_evaluate_fn(bel.mean, apply_fn, X_test[:ntest_per_batch], y_test[:ntest_per_batch],
                                           label="task1")
    miscl_result = {**overall_miscl_result, **current_miscl_result, **task1_miscl_result,}
    
    result = {
        "nll": nll_result,
        "miscl": miscl_result
    }
    
    return result


# Evaluation functions
nll_softmax = lambda logits, labels, int_labels: \
    optax.softmax_cross_entropy_with_integer_labels(logits, labels) if int_labels \
    else optax.softmax_cross_entropy(logits, labels)
ll_softmax = lambda logits, labels, int_labels: -nll_softmax(logits, labels, int_labels)
softmax_ll_il_clf_eval_fn = partial(evaluate_function, label="ll",
                                    loss_fn=partial(ll_softmax, int_labels=True),)
softmax_nll_il_clf_eval_fn = partial(evaluate_function, label="nll",
                                     loss_fn=partial(nll_softmax, int_labels=True))

miscl_softmax = lambda logits, labels: \
    (logits.argmax(axis=-1) != labels).mean()
softmax_miscl_clf_eval_fn = partial(evaluate_function, loss_fn=miscl_softmax, label="miscl")

def softmax_clf_eval_fn(flat_params, apply_fn, X_test, y_test):
    nll = softmax_nll_il_clf_eval_fn(flat_params, apply_fn, X_test, y_test)
    miscl = softmax_miscl_clf_eval_fn(flat_params, apply_fn, X_test, y_test)
    result = {**nll, **miscl}
    
    return result
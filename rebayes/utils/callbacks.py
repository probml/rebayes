"""
Custom callbacks
"""

import jax
import jax.numpy as jnp


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

    slice_ix = jnp.arange(0, steps) + t

    X_test = jnp.take(X_test, slice_ix, axis=0, fill_value=0)
    y_test = jnp.take(y_test, slice_ix, axis=0, fill_value=0)

    # eval on all tasks test set
    yhat_test = apply_fn(bel.mean, X_test).squeeze()

    # De-normalise target variables
    y_test = y_test * ystd + ymean
    yhat_test = yhat_test.ravel() * ystd + ymean

    y_next = y.ravel() * ystd + ymean
    yhat_next = pred_obs.ravel() * ystd + ymean

    # Compute errors
    err_test = jnp.power(y_test - yhat_test, 2).mean()
    err = jnp.power(y_next - yhat_next, 2).mean()

    err_test = jnp.sqrt(err_test)
    err = jnp.sqrt(err)

    res = {
        "n-step-pred": yhat_test,
        "osa-error": err, # one-step ahead
        "nsa-error": err_test, # n-step ahead
    }

    return res

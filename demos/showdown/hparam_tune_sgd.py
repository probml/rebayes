import jax
import optax
import jax.numpy as jnp
import numpy as np
from functools import partial
from rebayes.sgd_filter import replay_sgd as rsgd
from bayes_opt import BayesianOptimization

def bbf(
    learning_rate,
    n_inner,
    # specify before training
    train,
    test,
    rank,
    params,
    callback,
    apply_fn,
    lossfn,
    dim_in,
    dim_out=1,
):
    n_inner = round(n_inner)
    X_train, y_train = train
    X_test, y_test = test

    test_callback_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    agent = rsgd.FifoSGD(
        lossfn,
        apply_fn=apply_fn,
        init_params=params,
        tx=optax.adam(learning_rate),
        buffer_size=rank,
        dim_features=dim_in,
        dim_output=dim_out,
        n_inner=n_inner,
    )

    bel, _ = agent.scan(X_train, y_train, progress_bar=False)
    metric = callback(bel, **test_callback_kwargs)["test"].item()
    
    isna = np.isnan(metric)
    metric = 10 if isna else metric
    return -metric


def create_optimizer(
    model,
    bounds,
    random_state,
    train,
    test,
    rank,
    lossfn,
    callback=None,
):
    key = jax.random.PRNGKey(random_state)
    apply_fn = model.apply

    _, dim_in = train[0].shape
    dim_out = train[1].shape
    if len(dim_out) > 1:
        dim_out = dim_out[1]
    else:
        dim_out = 1

    batch_init = jnp.ones((1, dim_in))
    params_init = model.init(key, batch_init)

    bbf_partial = partial(bbf,
        train=train,
        test=test,
        rank=rank,
        params=params_init,
        apply_fn=apply_fn,
        callback=callback,
        lossfn=lossfn,
        dim_in=dim_in,
        dim_out=dim_out,
    )

    optimizer = BayesianOptimization(
        f=bbf_partial,
        pbounds=bounds, 
        random_state=random_state,
        allow_duplicate_points=True,
    )
    
    return optimizer

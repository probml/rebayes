import os
import jax
import optax
import pickle
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from time import time
from functools import partial
from typing import Callable
from bayes_opt import BayesianOptimization
from jax.flatten_util import ravel_pytree
from rebayes.utils import datasets, uci_uncertainty_data

from rebayes.low_rank_filter import lrvga, lofi
from rebayes.sgd_filter import replay_sgd as rsgd

import hparam_tune_sgd as hp_sgd
import hparam_tune_ekf as hp_ekf
import hparam_tune_lofi as hp_lofi


class MLP(nn.Module):
    n_out: int
    activation: Callable = nn.elu
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_out)(x)
        return x


@partial(jax.jit, static_argnames=("apply_fn",))
def lossfn_rmse_fifo(params, counter, X, y, apply_fn):

    yhat = apply_fn(params, X).ravel()
    y = y.ravel()
    err = jnp.power(y - yhat, 2)

    loss = (err * counter).sum() / counter.sum()
    return loss


def train_callback(bel, *args, **kwargs):
    X_test, y_test = kwargs["X_test"], kwargs["y_test"]
    apply_fn = kwargs["apply_fn"]

    y_test = y_test.ravel()
    yhat = apply_fn(bel.mean, X_test).ravel()
    err = jnp.power(y_test - yhat.ravel(), 2).mean()
    err = jnp.sqrt(err)
    
    res = {
        "test": err.mean(),
    }
    return res


def apply_fn_sgd(params, x, model):
    return model.apply(params, x)


def apply_main(flat_params, x, model, unflatten_fn):
    return model.apply(unflatten_fn(flat_params), x)


def eval_callback_main(bel, pred, t, X, y, bel_pred, ymean, ystd, **kwargs):
    X_test, y_test = kwargs["X_test"], kwargs["y_test"]
    apply_fn = kwargs["apply_fn"]    

    yhat_test = apply_fn(bel.mean, X_test).squeeze()
    
    # Compute errors
    y_test = y_test * ystd + ymean
    yhat_test = yhat_test.ravel() * ystd + ymean
    
    y_next = y.ravel() * ystd + ymean
    yhat_next = pred.ravel() * ystd + ymean
    
    err_test = jnp.power(y_test - yhat_test, 2).mean()
    err = jnp.power(y_next - yhat_next, 2).mean()
    
    err_test = jnp.sqrt(err_test)
    err = jnp.sqrt(err)
    
    res = {
        "test": err_test,
        "osa-error": err,
    }
    return res


def prepare_dataset(train, test, n_warmup=1000, n_test_warmup=300, normalise_features=True, normalise_target=True):
    data, csts = datasets.showdown_preprocess(train, test, n_warmup=n_warmup, n_test_warmup=n_test_warmup,
                                            normalise_features=normalise_features, normalise_target=normalise_target)
    data = jax.tree_map(jnp.nan_to_num, data)

    ymean = csts["ymean"]
    ystd = csts["ystd"]

    warmup_train = data["warmup_train"]
    warmup_test = data["warmup_test"]
    X_learn, y_learn = data["train"]
    X_test, y_test = data["test"]


    data = {
        "train": (X_learn, y_learn),
        "test": (X_test, y_test),
        "warmup": warmup_test,
        "ymean": ymean,
        "ystd": ystd,
    }
    return data


def tree_to_cpu(tree):
    return jax.tree_map(np.array, tree)


def get_subtree(tree, key):
    return jax.tree_map(lambda x: x[key], tree, is_leaf=lambda x: key in x)


def fwd_link_main(mean, bel, x, model, reconstruct_fn):
    params = reconstruct_fn(mean)
    means = model.apply(params, x).ravel()
    std = bel.sigma
    return means, std ** 2


def log_prob_main(mean, bel, x, y, fwd_link):
    yhat, std = fwd_link(mean, bel, x)
    std = jnp.sqrt(std)
    
    logp = distrax.Normal(yhat, std).log_prob(y).sum()
    return logp


def rmae_callback(bel, *args, **kwargs):
    X_test, y_test = kwargs["X_test"], kwargs["y_test"]
    apply_fn = kwargs["apply_fn"]
    
    yhat = apply_fn(bel.mean, X_test).squeeze()
    err = jnp.abs(y_test - yhat.ravel())
    
    res = {
        "test": err.mean(),
    }
    return res


def eval_sgd_agent(
        train, test, apply_fn, callback, agent, bel_init
):
    X_test, y_test = test
    X_train, y_train = train

    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    bel, losses = agent.scan(
        X_train, y_train, bel=bel_init, callback=callback, **test_kwargs
    )

    return bel, losses


def eval_lofi_agent(
    train, test, optimizer, flat_params, params_lofi, apply_fn, method, callback,
    progress_bar=False
):
    X_test, y_test = test
    X_train, y_train = train

    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    hparams = hp_lofi.get_best_params(None, optimizer)
    agent = hp_lofi.build_estimator(flat_params, hparams, params_lofi, apply_fn, method=method)
    bel, output = agent.scan(
        X_train, y_train, callback=callback, progress_bar=progress_bar, **test_kwargs
    )

    return bel, output


def eval_ekf_agent(
    train, test, optimizer, flat_params, apply_fn, method, callback,
    progress_bar=False
):
    X_test, y_test = test
    X_train, y_train = train

    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    hparams = hp_ekf.get_best_params(None, optimizer, method=method)
    agent = hp_ekf.build_estimator(flat_params, hparams, None, apply_fn, method=method)
    bel, output = agent.scan(
        X_train, y_train, callback=callback, progress_bar=progress_bar, **test_kwargs
    )

    return bel, output


def eval_lrvga(
    train, test, optimizer, apply_fn, callback,
    key, model, fwd_link, log_prob, dim_rank, n_outer, n_inner,
    n_samples=30, progress_bar=False
):
    X_train, y_train = train
    X_test, y_test = test

    if type(optimizer) == dict:
        optimizer = optimizer.copy()
    else:
        optimizer = optimizer.max["params"].copy()

    hparams = jax.tree_map(np.exp, optimizer)
    bel_init, _ = lrvga.init_lrvga(key, model, X_train, dim_rank, **hparams)

    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    agent = lrvga.LRVGA(fwd_link, log_prob, n_samples=n_samples, n_outer=n_outer, n_inner=n_inner)
    bel, output = agent.scan(
        X_train, y_train, callback=callback, progress_bar=progress_bar, bel=bel_init, **test_kwargs
    )

    return bel, output


def train_sgd_agent(params, model, method, datasets,
                    pbounds, train_callback, eval_callback,
                    optimizer_eval_kwargs, rank,
                    random_state=314):
    train, test, warmup = datasets
    X_train, y_train = train
    X_test, y_test = test

    part_apply_fn_sgd = partial(apply_fn_sgd, model=model)

    optimizer = hp_sgd.create_optimizer(
        model, pbounds, random_state, warmup, test, rank, lossfn_rmse_fifo,
        train_callback,
    )

    optimizer.maximize(
        **optimizer_eval_kwargs
    )

    best_hparams = optimizer.max["params"]
    learning_rate = best_hparams["learning_rate"]
    n_inner = round(best_hparams["n_inner"])


    bel_init = rsgd.FifoTrainState.create(
        apply_fn=part_apply_fn_sgd,
        params=params,
        tx=optax.adam(learning_rate=learning_rate),
        buffer_size=rank,
        dim_features=X_train.shape[1],
        dim_output=1,
    )

    agent = rsgd.FSGD(lossfn_rmse_fifo, n_inner=n_inner)

    time_init = time()
    bel, output = eval_sgd_agent(
        train, test, part_apply_fn_sgd, eval_callback, agent, bel_init,
    )
    time_end = time()

    res = {
        "method": method,
        "hparams": optimizer.max,
        "output": output,
        "beliefs": bel.replace(apply_fn=None, tx=None),
        "running_time": time_end - time_init,
    }

    return res


def train_ekf_agent(params, model, method, datasets,
                    pbounds, train_callback, eval_callback,
                    optimizer_eval_kwargs,
                    random_state=314):
    train, test, warmup = datasets
    flat_params, _ = ravel_pytree(params)
    n_params = len(flat_params)

    X_train, y_train = train
    X_test, y_test = test

    _, num_features = X_train.shape

    optimizer, apply_fn, _ = hp_ekf.create_optimizer(
        model, pbounds, random_state, warmup, test, train_callback, method=method
    )

    optimizer.maximize(
        **optimizer_eval_kwargs,
    )

    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    hparams = hp_ekf.get_best_params(n_params, optimizer, method=method)
    agent = hp_ekf.build_estimator(flat_params, hparams, None, apply_fn, method=method)
    bel, output = agent.scan(
        X_train, y_train, callback=eval_callback, progress_bar=False, **test_kwargs
    )

    res = {
        "method": method,
        "hparams": optimizer.max,
        "output": output,
        "beliefs": bel,
    }

    return res, apply_fn


def train_lofi_agent(params, params_lofi, model, method, dataset,
                     pbounds, train_callback, eval_callback,
                     optimizer_eval_kwargs,
                     random_state=314):
    train, test, warmup = dataset
    flat_params, _ = ravel_pytree(params)
    n_params = len(flat_params)

    X_train, y_train = train
    X_test, y_test = test

    optimizer, apply_fn, _ = hp_lofi.create_optimizer(
        model, pbounds, random_state, warmup, test, params_lofi, train_callback, method=method
    )

    optimizer.maximize(
        **optimizer_eval_kwargs
    )

    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    hparams = hp_lofi.get_best_params(n_params, optimizer)
    agent = hp_lofi.build_estimator(flat_params, hparams, params_lofi, apply_fn, method=method)

    time_init = time()
    bel, output = agent.scan(
        X_train, y_train, callback=eval_callback, progress_bar=False, **test_kwargs
    )
    time_end = time()

    res = {
        "method": method,
        "hparams": optimizer.max,
        "output": output,
        "beliefs": bel,
        "running_time": time_end - time_init,
    }

    return res, apply_fn, hparams


def train_lrvga_agent(key, apply_fn, model, dataset, dim_rank, n_inner, n_outer,
                      pbounds, eval_callback, fwd_link,
                      optimizer_eval_kwargs, random_state=314):
    method = "lrvga"
    train, test, warmup = dataset

    X_train, y_train = train
    X_test, y_test = test

    log_prob = partial(log_prob_main, fwd_link=fwd_link)
    def bbf(std, sigma2, eps, train, test, n_inner, n_outer):
        X_train, y_train = train
        X_test, y_test = test
        
        std = np.exp(std)
        sigma2 = np.exp(sigma2)
        eps = np.exp(eps)
            
        hparams = {
            "std": std,
            "sigma2": sigma2,
            "eps": eps,
        }
        
        bel_init, _ = lrvga.init_lrvga(key, model, X_train, dim_rank, **hparams)
        agent = lrvga.LRVGA(fwd_link, log_prob, n_samples=30, n_outer=n_outer, n_inner=n_inner)
        bel, _ = agent.scan(X_train, y_train, progress_bar=False, bel=bel_init)
        
        metric = jnp.abs(agent.predict_obs(bel, X_test) - y_test).mean()
        isna = np.isnan(metric)
        metric = 1000 if isna else metric
        return -metric
    
    pbbf = partial(bbf, train=warmup, test=test, n_outer=n_outer, n_inner=n_inner)

    optimizer = BayesianOptimization(
        f=pbbf,
        pbounds=pbounds,
        random_state=random_state,
        allow_duplicate_points=True,
    )

    optimizer.maximize(
        **optimizer_eval_kwargs
    )

    hparams = jax.tree_map(np.exp, optimizer.max["params"])
    bel_init, _ = lrvga.init_lrvga(key, model, X_train, dim_rank, **hparams)

    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    agent = lrvga.LRVGA(fwd_link, log_prob, n_samples=30, n_outer=n_outer, n_inner=n_inner)
    bel, output = agent.scan(
        X_train, y_train, callback=eval_callback, progress_bar=False, bel=bel_init, **test_kwargs
    )

    res = {
        "method": method,
        "hparams": optimizer.max,
        "output": output,
        "beliefs": bel,
    }

    return res


def store_results(results, name, path):
    path = os.path.join(path, name)
    filename = f"{path}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def train_agents(key, dim_rank, dataset_name, path, output_path, ix):
    """
    TODO: divide script into memory-aware agents and memory-agnostic agents.
    Memory aware: LoFi, L-RVGA, ORFit
    Memory-agnostic: EKF-FC, EKF-FD, EFK-VD.
    """
    # TODO: Move to individual script
    res = uci_uncertainty_data.load_data(path, ix)
    dataset = res["dataset"]
    # Simple offset of NaNs to zero-values
    dataset = jax.tree_map(jnp.nan_to_num, dataset)
    X_train, _ = dataset["train"]

    # Train, test, warmup
    dataset = dataset["train"], dataset["test"], dataset["train"][:500]

    ymean = res["ymean"]
    ystd = res["ystd"]
    eval_callback = partial(eval_callback_main, ymean=ymean, ystd=ystd)

    _, dim_in = X_train.shape
    dim_out = 1
    model = MLP(dim_out, activation=nn.relu)
    params = model.init(key, jnp.ones((1, dim_in)))
    params_flat, reconstruct_fn =  ravel_pytree(params)

    optimizer_eval_kwargs = {
        "init_points": 10,
        "n_iter": 15,
    }

    pbounds = {
        "log_init_cov": (-5, 0.0),
        "dynamics_weights": (0, 1.0),
        "log_emission_cov": (-7, 0.0),
        "dynamics_log_cov": (-7, 0.0),
    }

    pbounds_lofi = pbounds.copy()
    pbounds_lofi.pop("dynamics_log_cov")
    pbounds_lofi["dynamics_covariance"] = None

    pbounds_lrvga = {
        "std": (-0.34, 0.0),
        "sigma2": (-4, 0.0),
        "eps": (-10, -4),
    }

    # -------------------------------------------------------------------------
    # Extended Kalman Filter
    methods = ["fdekf", "vdekf", "fcekf"]
    for method in methods:
        res, apply_fn = train_ekf_agent(
            params, model, method, dataset, pbounds,
            train_callback, eval_callback,
            optimizer_eval_kwargs,
        )

        metric_final = res["output"]["test"][-1]
        print(method)
        print(f"{metric_final:=0.4f}")
        print("-" * 80)
        store_results(res, f"{dataset_name}_{method}_{ix}", output_path)

    # -------------------------------------------------------------------------
    # Low-rank filter
    methods = ["lofi_orth", "lofi"]
    params_lofi = lofi.LoFiParams(
        memory_size=dim_rank,
        sv_threshold=0,
        steady_state=True,
    )
    for method in methods:
        res, apply_fn, hparams = train_lofi_agent(
            params, params_lofi, model, method, dataset, pbounds_lofi,
            train_callback, eval_callback,
            optimizer_eval_kwargs,
        )

        metric_final = res["output"]["test"][-1]
        print(method)
        print(f"{metric_final:=0.4f}")
        print("-" * 80)
        store_results(res, f"{dataset_name}_{method}_{ix}", output_path)


    # -------------------------------------------------------------------------
    # Low-rank variational Gaussian approximation (LRVGA)

    optimizer_eval_kwargs = {
        "init_points": 5,
        "n_iter": 0,
    }
    n_outer = 6
    n_inner = 4
    fwd_link = partial(fwd_link_main, model=model, reconstruct_fn=reconstruct_fn)
    res = train_lrvga_agent(
        key, apply_fn, model, dataset, dim_rank=dim_rank, n_inner=n_inner, n_outer=n_outer,
        pbounds=pbounds_lrvga, eval_callback=eval_callback,  fwd_link=fwd_link,
        optimizer_eval_kwargs=optimizer_eval_kwargs,
        random_state=random_state,
    ) 
    method = res["method"]

    metric_final = res["output"]["test"][-1]
    print(method)
    print(f"{metric_final:=0.4f}")
    print("-" * 80)
    store_results(res, f"{dataset_name}_{method}_{ix}", output_path)


if __name__ == "__main__":
    from itertools import product
    # TOODO: change to $REBAYES_OUTPUT, $REBAYES_DATASET
    output_path = "/home/gerardoduran/documents/rebayes/demos/showdown/output/checkpoints"
    dataset_path = "/home/gerardoduran/documents/external/DropoutUncertaintyExps/UCI_Datasets"

    # _, dataset_name = sys.argv
    
    random_state = 314
    key = jax.random.PRNGKey(314)

    num_partitions = 1
    partitions = range(num_partitions)
    datasets = [
        "bostonHousing", "concrete", "energy", "kin8nm", "naval-propulsion-plant",
        "power-plant", "wine-quality-red", "yacht"
    ]

    dim_ranks = [1, 2, 5, 10, 20, 50]
    ix = 0
    for i, (dataset_name, dim_rank) in enumerate(product(datasets, dim_ranks)):
        keyv = jax.random.fold_in(key, i)
        data_path = os.path.join(dataset_path, dataset_name, "data")
        dataset_name_store = f"{dataset_name}_rank{dim_rank:02}"
        print(f"Fitting {dataset_name_store} --- {ix}")
        train_agents(keyv, dim_rank, dataset_name_store, data_path, output_path, ix)
        print("\n" * 2)

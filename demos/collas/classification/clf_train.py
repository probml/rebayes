from functools import partial
from pathlib import Path
import pickle
from time import time
from typing import Callable

import matplotlib.pyplot as plt
import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax_tqdm import scan_tqdm
from tqdm import trange
import jax_dataloader.core as jdl

from rebayes.utils.callbacks import cb_clf_discrete_tasks


def eval_agent_stationary(
    model_init_fn: Callable,
    dataset_load_fn: Callable,
    optimizer_dict: dict,
    eval_callback: Callable,
    n_iter: int=20,
    key: int=0
) -> dict:
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    dataset = dataset_load_fn()
    model = model_init_fn(0)
    agent, init_cov = optimizer_dict["agent"], optimizer_dict["init_cov"]
    X_test, y_test = dataset["test"]
    test_kwargs = {"agent": agent, "X_test": X_test, "y_test": y_test, 
                   "apply_fn": model["apply_fn"], "key": key}
    
    @scan_tqdm(n_iter)
    def _step(_, key):
        keys = jr.split(key)
        dataset = dataset_load_fn(key=keys[0])
        X_train, y_train = dataset["train"]
        
        model = model_init_fn(keys[1])
        _, result = agent.scan(model["flat_params"], init_cov, X_train, y_train,
                               callback=eval_callback, **test_kwargs)
        
        return None, result

    carry = None
    start_time = time()
    keys = jr.split(subkey, n_iter)
    _, result = jax.block_until_ready(lax.scan(_step, carry, keys))
    runtime = time() - start_time
    result["runtime"] = runtime

    return result


def eval_agent_nonstationary(
    model_init_fn: Callable,
    dataset_load_fn: Callable,
    optimizer_dict: dict,
    eval_callback: Callable,
    n_iter: int=20,
    key: int=0,
    **kwargs,
) -> dict:
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    agent, init_cov = optimizer_dict["agent"], optimizer_dict["init_cov"]
    result = None
    start_time = time()
    for _ in trange(n_iter, desc='Evaluating agent...'):
        # Load dataset with random permutation and random shuffle
        key1, key2, subkey = jr.split(subkey, 3)
        dataset = dataset_load_fn(key=key1)
        Xtr, Ytr, *_ = dataset["train"]
        Xte, Yte, *_ = dataset["test"]
        train_ds = jdl.ArrayDataset(Xtr, Ytr)
        train_loader = jdl.DataLoaderJax(
            train_ds, batch_size=kwargs["ntrain_per_task"], shuffle=False, 
            drop_last=False
        )
        model_dict = model_init_fn(key2)
        init_mean, apply_fn = model_dict['flat_params'], model_dict['apply_fn']
        
        test_kwargs = {
            'X_test': Xte,
            'y_test': Yte,
            'ntest_per_batch': kwargs["ntest_per_task"],
            'apply_fn': apply_fn,
        }
        
        _, curr_result = agent.scan_dataloader(
            init_mean,
            init_cov,
            train_loader, 
            callback=eval_callback,
            callback_at_end=False,
            **test_kwargs
        )
        curr_result = jax.tree_map(lambda *xs: jnp.concatenate(xs), *curr_result)
        if result is None:
            result = curr_result
        else:
            result = jax.tree_map(lambda x, y: jnp.array([*x, y]), result, curr_result)
    runtime = time() - start_time
    result["runtime"] = runtime
    
    return result


# ------------------------------------------------------------------------------
# Save and Plot Results

def store_results(results, name, path):
    path = Path(path, name)
    filename = f"{path}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f) 


def plot_results(results, name, path, ylim, ax=None, title=''):
    if ax is None:
        fig, ax = plt.subplots()
    path = Path(path, name)
    filename = f"{path}.pdf"
    plt.figure(figsize=(10, 5))
    for key, val in results.items():
        mean, std = val['mean'], val['std']
        ax.plot(mean, label=key)
        ax.fill_between(
            jnp.arange(mean.shape[0]),
            mean - std,
            mean + std,
            alpha=0.3
        )
    ax.set_xlabel('Number of training examples seen')
    ax.set_ylabel(title)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend()
    fig.savefig(filename)
    
    return ax


# ------------------------------------------------------------------------------
# Deprecated

# def rotating_mnist_eval_agent(
#     train, test, apply_fn, callback, agent, bel_init=None, n_iter=20, n_steps=500,
#     min_angle=0, max_angle=360
# ):      
#     X_train, y_train = train
#     X_test, y_test = test
#     test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
#     gradually_rotating_angles = jnp.linspace(min_angle, max_angle, n_steps)
    
#     @scan_tqdm(n_iter)
#     def _step(_, i):
#         key = jr.PRNGKey(i)
#         indx = jr.choice(key, len(X_train), (n_steps,))
#         X_curr, y_curr = X_train[indx], y_train[indx]
#         # Rotate the images
#         X_curr = rotate_mnist_dataset(X_curr, gradually_rotating_angles)
#         _, result = agent.scan(X_curr, y_curr, callback=callback, **test_kwargs, bel=bel_init)
        
#         return None, result

#     carry = None
#     start_time = time()
#     _, output = lax.scan(_step, carry, jnp.arange(n_iter))
#     # mean, std = jax.block_until_ready(res.mean(axis=0)), jax.block_until_ready(res.std(axis=0))
#     runtime = time() - start_time

#     return output, runtime
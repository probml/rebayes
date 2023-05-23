from typing import Sequence
from functools import partial
from pathlib import Path
import pickle
from time import time

import matplotlib.pyplot as plt
from flax import linen as nn
import jax
from jax import jit, vmap, lax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import optax
from jax_tqdm import scan_tqdm
from tqdm import trange
import jax_dataloader.core as jdl

from rebayes.datasets.rotating_permuted_mnist_data import (
    generate_rotating_mnist_dataset,
    rotate_mnist_dataset
)
from rebayes.utils.utils import (
    get_mlp_flattened_params
)
from rebayes.low_rank_filter.lofi_core import _jacrev_2d


# ------------------------------------------------------------------------------
# NN Models

class CNN(nn.Module):
    output_dim: int = 10
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
    

def init_model(key=0, type='cnn', features=(400, 400, 10), classification=True, rescale=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    input_dim = [1, 28, 28, 1]
    model_dim = [input_dim, *features]
    if type == 'cnn':
        if classification:
            model = CNN()
        else:
            model = CNN(output_dim=1)
        params = model.init(key, jnp.ones(input_dim))['params']
        flat_params, unflatten_fn = ravel_pytree(params)
        apply_fn = lambda w, x: model.apply({'params': unflatten_fn(w)}, x).ravel()

        emission_mean_function = apply_fn
    elif type == 'mlp':
        if rescale:
            model, flat_params, _, apply_fn = get_mlp_flattened_params(model_dim, key)
        else:
            model = MLP(features)
            params = model.init(key, jnp.ones(input_dim))['params']
            flat_params, unflatten_fn = ravel_pytree(params)
            apply_fn = lambda w, x: model.apply({'params': unflatten_fn(w)}, x).ravel()
            
    else:
        raise ValueError(f'Unknown model type: {type}')
    
    model_dict = {
        'model': model,
        'flat_params': flat_params,
        'apply_fn': apply_fn,
    }
    
    if classification:
        if features[-1] == 1:
            # Binary classification
            sigmoid_fn = lambda w, x: jnp.clip(jax.nn.sigmoid(apply_fn(w, x)), 1e-4, 1-1e-4).ravel()
            emission_mean_function = lambda w, x: sigmoid_fn(w, x)
            emission_cov_function = lambda w, x: sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x))
        else:
            # Multiclass classification
            emission_mean_function=lambda w, x: jax.nn.softmax(apply_fn(w, x))
            def emission_cov_function(w, x):
                ps = emission_mean_function(w, x)
                return jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * jnp.eye(len(ps)) # Add diagonal to avoid singularity
            
            def replay_emission_cov_function(w, w_lin, x):
                m_Y = lambda w: emission_mean_function(w, x)
                H = _jacrev_2d(m_Y, w_lin)
                ps = jnp.atleast_1d(m_Y(w_lin)) + H @ (w - w_lin)
                return jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * jnp.eye(len(ps)) # Add diagonal to avoid singularity
            model_dict["replay_emission_cov_function"] = replay_emission_cov_function
        model_dict['emission_mean_function'] = emission_mean_function
        model_dict['emission_cov_function'] = emission_cov_function
    else:
        # Regression
        emission_mean_function = apply_fn
        model_dict['emission_mean_function'] = emission_mean_function
    
    return model_dict


# ------------------------------------------------------------------------------
# Callback Functions

@partial(jit, static_argnums=(1,4,))
def evaluate_function(flat_params, apply_fn, X_test, y_test, loss_fn):
    @jit
    def evaluate(label, image):
        image = image.reshape((1, 28, 28, 1))
        logits = apply_fn(flat_params, image).ravel()
        loss = loss_fn(logits, label)
        
        return loss
    evals = vmap(evaluate, (0, 0))(y_test, X_test)
    
    return evals.mean()
    

def eval_callback(bel, *args, evaluate_fn, nan_val=-1e8, **kwargs):
    X, y, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    eval = evaluate_fn(bel.mean, apply_fn, X, y)
    if isinstance(eval, dict):
        eval = {k: jnp.where(jnp.isnan(v), nan_val, v) for k, v in eval.items()}
    else:
        eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    
    return eval


def eval_nlpd_mc_eval_callback(bel, pred_obs, t, *args, evaluate_fn, nan_val=-1e8, **kwargs):
    agent, X, y, apply_fn = \
        kwargs["agent"], kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    key = jax.random.fold_in(kwargs["key"], t)
    
    if evaluate_fn is None:
        eval = {}
    else:
        eval = evaluate_fn(bel.mean, apply_fn, X, y)
        if not isinstance(eval, dict):
            eval = {"eval": eval}
    nlpd = agent.nlpd_mc(bel, key, X, y).mean()
    eval["nlpd"] = nlpd
    eval = {k: jnp.where(jnp.isnan(v), nan_val, v) for k, v in eval.items()}
    
    return eval


def eval_lpd_mc_callback(bel, pred_obs, t, *args, nan_val=-1e8, **kwargs):
    agent, X, y, apply_fn = \
        kwargs["agent"], kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    key = jax.random.fold_in(kwargs["key"], t)
    nlpd = agent.nlpd_mc(bel, key, X, y).mean()
    lpd = jnp.where(jnp.isnan(nlpd), nan_val, -nlpd)
    
    return lpd


def eval_nlpd_mc_callback(bel, pred_obs, t, *args, nan_val=-1e8, glm_predictive=False, **kwargs):
    lpd = eval_lpd_mc_callback(bel, pred_obs, t, *args, nan_val=nan_val, glm_predictive=glm_predictive, **kwargs)
    
    return -lpd


def osa_eval_callback(bel, y_pred, t, X, y, bel_pred, evaluate_fn, nan_val=-1e8, **kwargs):
    eval = evaluate_fn(y_pred, y)
    eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    
    return eval


def osa_nlpd_mc_callback(bel, y_pred, t, X, y, bel_pred, **kwargs):
    agent, key = kwargs["agent"], kwargs["key"]
    
    nlpd = agent.nlpd_mc(key, bel, X, y)
    
    result = {
        "nlpd": nlpd,
    }
    
    return result


def per_batch_callback(i, bel_pre_update, bel, batch, evaluate_fn, **kwargs):
    X_test, y_test, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    ntest_per_batch = kwargs["ntest_per_batch"]
    
    prev_test_batch, curr_test_batch = i*ntest_per_batch, (i+1)*ntest_per_batch
    curr_X_test, curr_y_test = X_test[prev_test_batch:curr_test_batch], y_test[prev_test_batch:curr_test_batch]
    cum_X_test, cum_y_test = X_test[:curr_test_batch], y_test[:curr_test_batch]
    
    overall_accuracy = evaluate_fn(bel.mean, apply_fn, cum_X_test, cum_y_test)
    current_accuracy = evaluate_fn(bel.mean, apply_fn, curr_X_test, curr_y_test)
    task1_accuracy = evaluate_fn(bel.mean, apply_fn, X_test[:ntest_per_batch], y_test[:ntest_per_batch])
    result = {
        'overall': overall_accuracy,
        'current': current_accuracy,
        'first_task': task1_accuracy,
    }
    
    return result


def window_callback_loss(bel, y_pred, t, X, y, bel_pred, loss_fn, nan_val=-1e8, window_size=50, **kwargs):
    X_test, y_test, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    test_idx = jnp.arange(window_size) + t - window_size // 2
    X_test, y_test = X_test[test_idx], y_test[test_idx]
        
    eval = -evaluate_function(bel.mean, apply_fn, X_test, y_test, loss_fn)
    if isinstance(eval, dict):
        eval = {k: jnp.where(jnp.isnan(v), nan_val, v) for k, v in eval.items()}
    else:
        eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    
    return eval


def window_callback_eval(bel, y_pred, t, X, y, bel_pred, evaluate_fn, nan_val=-1e8, window_size=50, **kwargs):
    X_test, y_test, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    test_idx = jnp.arange(window_size) + t - window_size // 2
    X_test, y_test = X_test[test_idx], y_test[test_idx]
        
    eval = evaluate_fn(bel.mean, apply_fn, X_test, y_test)
    if isinstance(eval, dict):
        eval = {k: jnp.where(jnp.isnan(v), nan_val, v) for k, v in eval.items()}
    else:
        eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    
    return eval


# MNIST
def mnist_evaluate_nll(flat_params, apply_fn, X_test, y_test):
    nll = evaluate_function(flat_params, apply_fn, X_test, y_test, optax.softmax_cross_entropy_with_integer_labels)
    
    return nll

def mnist_evaluate_ll(flat_params, apply_fn, X_test, y_test):
    nll = mnist_evaluate_nll(flat_params, apply_fn, X_test, y_test)
    
    return -nll


def mnist_evaluate_miscl(flat_params, apply_fn, X_test, y_test):
    acc_fn = lambda logits, label: (logits.argmax(axis=-1) == label).mean()
    acc = evaluate_function(flat_params, apply_fn, X_test, y_test, acc_fn)
    
    return 1.0 - acc


def mnist_evaluate_nll_and_miscl(flat_params, apply_fn, X_test, y_test):
    nll = mnist_evaluate_nll(flat_params, apply_fn, X_test, y_test)
    miscl = mnist_evaluate_miscl(flat_params, apply_fn, X_test, y_test)
    
    result = {
        "nll": nll,
        "miscl": miscl,
    }
    
    return result


# Split-MNIST
def smnist_evaluate_log_likelihood(flat_params, apply_fn, X_test, y_test):
    nll = evaluate_function(flat_params, apply_fn, X_test, y_test, optax.sigmoid_binary_cross_entropy)
    
    return -nll


def smnist_evaluate_accuracy(flat_params, apply_fn, X_test, y_test):
    acc_fn = lambda logits, label: (jnp.round(jax.nn.sigmoid(logits)) == label).mean()
    acc = evaluate_function(flat_params, apply_fn, X_test, y_test, acc_fn)
    
    return acc


# ------------------------------------------------------------------------------
# Model Evaluation

def mnist_eval_agent(
    train, test, apply_fn, callback, agent, bel_init=None, n_iter=20, n_steps=500, key=0
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    X_train, y_train = train
    X_test, y_test = test
    test_kwargs = {"agent": agent, "X_test": X_test, "y_test": y_test, 
                   "apply_fn": apply_fn, "key": key}
    
    @scan_tqdm(n_iter)
    def _step(_, i):
        key = jr.PRNGKey(i)
        indx = jr.choice(key, len(X_train), (n_steps,))
        X_curr, y_curr = X_train[indx], y_train[indx]
        _, result = agent.scan(X_curr, y_curr, callback=callback, **test_kwargs, bel=bel_init)
        
        return None, result

    carry = None
    start_time = time()
    _, output = lax.scan(_step, carry, jnp.arange(n_iter))
    # mean, std = jax.block_until_ready(res.mean(axis=0)), jax.block_until_ready(res.std(axis=0))
    runtime = time() - start_time

    return output, runtime


def rotating_mnist_eval_agent(
    train, test, apply_fn, callback, agent, bel_init=None, n_iter=20, n_steps=500,
    min_angle=0, max_angle=360
):      
    X_train, y_train = train
    X_test, y_test = test
    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    gradually_rotating_angles = jnp.linspace(min_angle, max_angle, n_steps)
    
    @scan_tqdm(n_iter)
    def _step(_, i):
        key = jr.PRNGKey(i)
        indx = jr.choice(key, len(X_train), (n_steps,))
        X_curr, y_curr = X_train[indx], y_train[indx]
        # Rotate the images
        X_curr = rotate_mnist_dataset(X_curr, gradually_rotating_angles)
        _, result = agent.scan(X_curr, y_curr, callback=callback, **test_kwargs, bel=bel_init)
        
        return None, result

    carry = None
    start_time = time()
    _, output = lax.scan(_step, carry, jnp.arange(n_iter))
    # mean, std = jax.block_until_ready(res.mean(axis=0)), jax.block_until_ready(res.std(axis=0))
    runtime = time() - start_time

    return output, runtime


def nonstationary_mnist_callback(bel, pred_obs, t, x, y, bel_pred, i, 
                                 nll_loss_fn=None, miscl_loss_fn=None, **kwargs):
    if nll_loss_fn is None:
        nll_loss_fn = lambda logits, label: optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
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
    curr_X_test, curr_y_test = X_test[prev_test_batch:curr_test_batch], y_test[prev_test_batch:curr_test_batch]
    cum_X_test, cum_y_test = X_test[:curr_test_batch], y_test[:curr_test_batch]
    
    overall_nll_result = nll_evaluate_fn(bel.mean, apply_fn, cum_X_test, cum_y_test)
    current_nll_result = nll_evaluate_fn(bel.mean, apply_fn, curr_X_test, curr_y_test)
    task1_nll_result = nll_evaluate_fn(bel.mean, apply_fn, X_test[:ntest_per_batch], y_test[:ntest_per_batch])
    nll_result = {
        'overall': overall_nll_result,
        'current': current_nll_result,
        'task1': task1_nll_result,
    }
    
    overall_miscl_result = miscl_evaluate_fn(bel.mean, apply_fn, cum_X_test, cum_y_test)
    current_miscl_result = miscl_evaluate_fn(bel.mean, apply_fn, curr_X_test, curr_y_test)
    task1_miscl_result = miscl_evaluate_fn(bel.mean, apply_fn, X_test[:ntest_per_batch], y_test[:ntest_per_batch])
    miscl_result = {
        'overall': overall_miscl_result,
        'current': current_miscl_result,
        'task1': task1_miscl_result,
    }
    
    result = {
        "nll": nll_result,
        "miscl": miscl_result
    }
    
    return result


def nonstationary_mnist_eval_agent(
    load_dataset_fn, 
    ntrain_per_task,
    ntest_per_task,
    apply_fn,
    agent,  
    bel=None,
    n_iter=10,
    nll_loss_fn=None, 
    miscl_loss_fn=None
):
    result = {
        "nll": {"overall": jnp.array([]), "current": jnp.array([]), "task1": jnp.array([])},
        "miscl": {"overall": jnp.array([]), "current": jnp.array([]), "task1": jnp.array([])},
    }
    for i in trange(n_iter, desc='Evaluating agent...'):
        # Load dataset with random permutation and random shuffle
        dataset = load_dataset_fn(key=i)
        (Xtr, Ytr), _, (Xte, Yte) = dataset.values()
        train_ds = jdl.ArrayDataset(Xtr, Ytr)
        train_loader = jdl.DataLoaderJax(
            train_ds, batch_size=ntrain_per_task, shuffle=False, drop_last=False
        )
        
        test_kwargs = {
            'X_test': Xte,
            'y_test': Yte,
            'ntest_per_batch': ntest_per_task,
            'apply_fn': apply_fn,
        }
        
        _, curr_result = agent.scan_dataloader(
            train_loader, 
            callback=partial(nonstationary_mnist_callback, nll_loss_fn=nll_loss_fn, miscl_loss_fn=miscl_loss_fn),
            bel=bel,
            callback_at_end=False,
            **test_kwargs
        )
        curr_result = jax.tree_map(lambda *xs: jnp.concatenate(xs), *curr_result)
        if result is None:
            result = curr_result
        else:
            result = jax.tree_map(lambda x, y: jnp.array([*x, y]), result, curr_result)
    
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
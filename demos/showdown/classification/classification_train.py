from typing import Sequence
from functools import partial
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
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

from rebayes.utils.avalanche import make_avalanche_data


# ------------------------------------------------------------------------------
# NN Models

class CNN(nn.Module):
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
        x = nn.Dense(features=10)(x)
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
    

def init_model(key=0, type='cnn', features=(400, 400, 10)):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    if type == 'cnn':
        model = CNN()
    elif type == 'mlp':
        model = MLP(features)
    else:
        raise ValueError(f'Unknown model type: {type}')
    
    params = model.init(key, jnp.ones([1, 28, 28, 1]))['params']
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: model.apply({'params': unflatten_fn(w)}, x).ravel()
    
    emission_mean_function=lambda w, x: jax.nn.softmax(apply_fn(w, x))
    def emission_cov_function(w, x):
        ps = emission_mean_function(w, x)
        return jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * jnp.eye(len(ps)) # Add diagonal to avoid singularity
    
    model_dict = {
        'model': model,
        'flat_params': flat_params,
        'apply_fn': apply_fn,
        'emission_mean_function': emission_mean_function,
        'emission_cov_function': emission_cov_function,
    }
    
    return model_dict


# ------------------------------------------------------------------------------
# Dataset Helper Functions

def load_mnist_dataset(fashion=False):
    """Load MNIST train and test datasets into memory."""
    dataset='mnist'
    if fashion:
        dataset='fashion_mnist'
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split='test[80%:]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test[:80%]', batch_size=-1))
    
    # Normalize pixel values
    for ds in [train_ds, val_ds, test_ds]:
        ds['image'] = jnp.float32(ds['image']) / 255.
    
    X_train, y_train = jnp.array(train_ds['image']), jnp.array(train_ds['label'])
    X_val, y_val = jnp.array(val_ds['image']), jnp.array(val_ds['label'])
    X_test, y_test = jnp.array(test_ds['image']), jnp.array(test_ds['label'])
    
    dataset = process_dataset(X_train, y_train, X_val, y_val, X_test, y_test)
        
    return dataset


def load_avalanche_mnist_dataset(avalanche_dataset, n_experiences, ntrain_per_dist, ntrain_per_batch, nval_per_batch, ntest_per_batch, seed=0, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    dataset = avalanche_dataset(n_experiences=n_experiences, seed=seed)
    Xtr, Ytr, Xte, Yte = make_avalanche_data(dataset, ntrain_per_dist, ntrain_per_batch, nval_per_batch + ntest_per_batch, key)
    Xtr, Xte = Xtr.reshape(-1, 1, 28, 28, 1), Xte.reshape(-1, 1, 28, 28, 1)
    Ytr, Yte = Ytr.ravel(), Yte.ravel()
    
    Xte_batches, Yte_batches = jnp.split(Xte, n_experiences), jnp.split(Yte, n_experiences)
    Xval_sets, Yval_sets = [batch[:nval_per_batch] for batch in Xte_batches], [batch[:nval_per_batch] for batch in Yte_batches]
    Xte_sets, Yte_sets = [batch[nval_per_batch:] for batch in Xte_batches], [batch[nval_per_batch:] for batch in Yte_batches]
    
    Xval, Yval = jnp.concatenate(Xval_sets), jnp.concatenate(Yval_sets)
    Xte, Yte = jnp.concatenate(Xte_sets), jnp.concatenate(Yte_sets)
    
    dataset = process_dataset(Xtr, Ytr, Xval, Yval, Xte, Yte)
    
    return dataset


def load_permuted_mnist_dataset(n_tasks, ntrain_per_task, nval_per_task, ntest_per_task, key=0, fashion=False):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
        
    dataset = load_mnist_dataset(fashion=fashion)
    
    def permute(x, idx):
        return x.ravel()[idx].reshape(x.shape)
    
    n_per_task = {'train': ntrain_per_task, 'val': nval_per_task, 'test': ntest_per_task}
    result = {data_type: ([], []) for data_type in ['train', 'val', 'test']}
    
    for _ in range(n_tasks):
        key, subkey = jr.split(key)
        perm_idx = jr.permutation(subkey, jnp.arange(28*28))
        permute_fn = partial(permute, idx=perm_idx)
        
        for data_type, data in dataset.items():
            key, subkey = jr.split(key)
            X, Y = data
            sample_idx = jr.choice(subkey, jnp.arange(len(X)), shape=(n_per_task[data_type],), replace=False)
            
            curr_X = vmap(permute_fn)(X[sample_idx])
            result[data_type][0].append(curr_X)
            
            curr_Y = Y[sample_idx]
            result[data_type][1].append(curr_Y)
    
    for data_type in ['train', 'val', 'test']:
        result[data_type] = (jnp.concatenate(result[data_type][0]), jnp.concatenate(result[data_type][1]))
            
    return result


def process_dataset(Xtr, Ytr, Xval, Yval, Xte, Yte):
    # Reshape data
    Xtr = Xtr.reshape(-1, 1, 28, 28, 1)
    Ytr_ohe = jax.nn.one_hot(Ytr, 10) # one-hot encode labels
    
    dataset = {
        'train': (Xtr, Ytr_ohe),
        'val': (Xval, Yval),
        'test': (Xte, Yte)
    }
    
    return dataset
    

# ------------------------------------------------------------------------------
# Callback Functions

@partial(jit, static_argnums=(1,4,))
def evaluate_function(flat_params, apply_fn, X_test, y_test, loss_fn):
    @jit
    def evaluate(label, image):
        image = image.reshape((1, 28, 28, 1))
        logits = apply_fn(flat_params, image).ravel()
        return loss_fn(logits, label)
    evals = vmap(evaluate, (0, 0))(y_test, X_test)
    
    return evals.mean()
    

def eval_callback(bel, *args, evaluate_fn, nan_val=-1e8, **kwargs):
    X, y, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    eval = evaluate_fn(bel.mean, apply_fn, X, y)
    eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    
    return eval


def osa_eval_callback(bel, y_pred, t, X, y, bel_pred, evaluate_fn, nan_val=-1e8, **kwargs):
    eval = evaluate_fn(y_pred, y)
    eval = jnp.where(jnp.isnan(eval), nan_val, eval)
    
    return eval


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
    train, test, apply_fn, callback, agent, bel_init=None, n_iter=5, n_steps=1_000,
):
    X_train, y_train = train
    X_test, y_test = test
    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    
    @scan_tqdm(n_iter)
    def _step(_, i):
        key = jr.PRNGKey(i)
        indx = jr.choice(key, len(X_train), (n_steps,))
        X_curr, y_curr = X_train[indx], y_train[indx]
        _, result = agent.scan(X_curr, y_curr, callback=callback, **test_kwargs, bel=bel_init)
        
        return result, result

    carry = jnp.zeros((n_steps,))
    _, res = lax.scan(_step, carry, jnp.arange(n_iter))
    mean, std = res.mean(axis=0), res.std(axis=0)

    return mean, std


def nonstationary_mnist_eval_agent(
    load_dataset_fn, 
    ntrain_per_task,
    ntest_per_task,
    apply_fn,
    agent, 
    callback, 
    bel=None,
    n_iter=5,
):
    overall_accs, current_accs, first_task_accs = [], [], []
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
        
        _, accs = agent.scan_dataloader(
            train_loader, 
            callback=callback,
            bel=bel,
            callback_at_end=True,
            **test_kwargs
        )
        overall_accs.append(jnp.array([res['overall'] for res in accs]))
        current_accs.append(jnp.array([res['current'] for res in accs]))
        first_task_accs.append(jnp.array([res['first_task'] for res in accs]))
    
    overall_accs, current_accs, task1_accs = \
        jnp.array(overall_accs).reshape((n_iter, -1)), \
            jnp.array(current_accs).reshape((n_iter, -1)), \
                jnp.array(task1_accs).reshape((n_iter, -1))

    result = {
        'overall': overall_accs.mean(axis=0),
        'overall-std': overall_accs.std(axis=0),
        'current': current_accs.mean(axis=0),
        'current-std': current_accs.std(axis=0),
        'first_task': task1_accs.mean(axis=0),
        'first_task-std': task1_accs.std(axis=0),
    }
    
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
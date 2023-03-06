from typing import Sequence

import tensorflow_datasets as tfds
from flax import linen as nn
import jax
from jax import jit, vmap, lax
import jax.numpy as jnp
import jax.random as jr
import optax

from avalanche.benchmarks.classic import SplitMNIST
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
    

# ------------------------------------------------------------------------------
# Dataset Helper Functions

def load_fmnist_datasets():
    """Load Fashion-MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('fashion_mnist')
    ds_builder.download_and_prepare()
    
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    
    # Normalize pixel values
    for ds in [train_ds, test_ds]:
        ds['image'] = jnp.float32(ds['image']) / 255.
        
    return train_ds, test_ds


def load_split_mnist_dataset(n_tasks=5, ntrain_per_task=200, ntest_per_task=500, key=0):
    """Load Split-MNIST train and test datasets into memory."""
    dataset = SplitMNIST(n_experiences=n_tasks, seed=0, return_task_id=True,
                         class_ids_from_zero_in_each_exp=True, fixed_class_order=range(10))
    nval_per_batch = ntest_per_task
    
    Xtr, Ytr, Xte, Yte =  make_avalanche_data(dataset, ntrain_per_task, ntrain_per_task, 2*ntest_per_task, key=key)
    Xtr, Xte = Xtr.reshape(-1, 1, 28, 28, 1), Xte.reshape(-1, 1, 28, 28, 1)
    Xte_batches, Yte_batches = jnp.split(Xte, n_tasks), jnp.split(Yte, n_tasks)
    Xval_sets, Yval_sets = [batch[:nval_per_batch] for batch in Xte_batches], [batch[:nval_per_batch] for batch in Yte_batches]
    Xte_sets, Yte_sets = [batch[nval_per_batch:] for batch in Xte_batches], [batch[nval_per_batch:] for batch in Yte_batches]

    Xval, Yval = jnp.concatenate(Xval_sets), jnp.concatenate(Yval_sets)
    Xte, Yte = jnp.concatenate(Xte_sets), jnp.concatenate(Yte_sets)

    return (Xtr, Ytr), (Xval, Yval), (Xte, Yte)


# ------------------------------------------------------------------------------
# Callback Functions

def evaluate_function(flat_params, apply_fn, X_test, y_test, loss_fn):
    @jit
    def evaluate(label, image):
        image = image.reshape((1, 28, 28, 1))
        logits = apply_fn(flat_params, image).ravel()
        return loss_fn(logits, label.ravel())
    evals = vmap(evaluate, (0, 0))(X_test, y_test)
    
    return evals.mean()
    

def eval_callback(bel, evaluate_fn, *args, **kwargs):
    X, y, apply_fn = kwargs["X_test"], kwargs["y_test"], kwargs["apply_fn"]
    eval = evaluate_fn(bel.mean, apply_fn, X, y)
    
    return eval


# MNIST
def mnist_evaluate_log_likelihood(flat_params, apply_fn, X_test, y_test):
    nll = evaluate_function(flat_params, apply_fn, X_test, y_test, optax.softmax_cross_entropy_with_integer_labels)
    
    return -nll


def mnist_evaluate_accuracy(flat_params, apply_fn, X_test, y_test):
    acc_fn = lambda logits, label: (logits.argmax(axis=-1) == label).mean()
    acc = evaluate_function(flat_params, apply_fn, X_test, y_test, acc_fn)
    
    return acc


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
    train, test, apply_fn, callback, agent, bel_init=None, key=0, n_iter=10, n_steps=1_000,
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    X_train, y_train = train
    X_test, y_test = test
    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    
    def _step(_, key):
        indx = jr.choice(key, len(X_train), (n_steps,))
        X_curr, y_curr = X_train[indx], y_train[indx]
        _, result = agent.scan(X_curr, y_curr, callback=callback, **test_kwargs, progress_bar=True, bel=bel_init)
        
        return result, result

    keys = jr.split(key, n_iter)
    carry = jnp.zeros((n_steps,))
    _, res = lax.scan(_step, carry, keys)
    mean, std = res.mean(axis=0), res.std(axis=0)

    return mean, std
    

# def smnist_eval_agent(
#     train, test, apply_fn, callback, agent, bel_init=None, n_iter=10,
# ): TODOTODO
    
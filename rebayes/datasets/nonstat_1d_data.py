"""
Nonstationary 1D data generation
"""
import jax
import jax.numpy as jnp
from copy import deepcopy

def make_1d_regression(
    key, n_train=100, n_test=100, sort_data=False, coef0=2.0, coef1=3.0, coef2=1.0
):  
    key_train, key_test = jax.random.split(key)
    keys_train = jax.random.split(key_train, n_train)
    keys_test = jax.random.split(key_test, n_test)
    minval, maxval = -0.5, 0.5

    def f(x):
        y = coef2 * x + 0.3 * jnp.sin(2.0 + coef1 * jnp.pi * x)
        return y
        
    @jax.vmap
    def gen(key):
        key_x, key_y = jax.random.split(key)
        x = jax.random.uniform(key_x, shape=(1,), minval=minval, maxval=maxval)
        if sort_data:
            x = jnp.sort(x)
        
        noise = jax.random.normal(key) * 0.02
        y = f(x) + noise
        return x, y
    
    X_train, y_train = gen(keys_train)
    X_test, y_test = gen(keys_test)

    X_eval = jnp.linspace(minval, maxval, 500)
    y_eval = f(X_eval)
    
    # Standardize dataset
    if True:
        X_train = (X_train - X_train.mean()) / X_train.std()
        y_train = (y_train - y_train.mean()) / y_train.std()
        X_test = (X_test - X_test.mean()) / X_test.std()
        y_test = (y_test - y_test.mean()) / y_test.std()
        X_eval = (X_eval - X_eval.mean()) / X_eval.std()
        y_eval = (y_eval - y_eval.mean()) / y_eval.std()

    train = (X_train, y_train)
    test = (X_test, y_test)
    eval  = (X_eval, y_eval)
    return train, test, eval


def make_coefs(key, n_dist):
    """
    Make c0, c1 distributions
    """
    key_slope, key_distort = jax.random.split(key)
    coefs = jax.random.uniform(key_distort, shape=(n_dist, 2), minval=-5, maxval=5)
    coef_slope = jax.random.uniform(key_slope, shape=(n_dist, 1), minval=-1.0, maxval=1.0)

    coefs = jnp.c_[coefs, coef_slope]
    return coefs


def sample_1d_regression_sequence(key, n_dist, n_train=100, n_test=100):
    key_coef, key_dataset = jax.random.split(key)
    keys_dataset = jax.random.split(key_dataset, n_dist)
    coefs = make_coefs(key, n_dist)
    
    @jax.vmap
    def vsample_dataset(key, coefs):
        train, test, eval = make_1d_regression(
            key, n_train, n_test, coef0=coefs[0], coef1=coefs[1], coef2=coefs[2]
        )
        return train, test, eval
    
    *collection, eval_dataset = vsample_dataset(keys_dataset, coefs)
    collection_flat = jax.tree_map(lambda x: x.reshape(-1, 1), collection)
    collection_train, collection_test = collection_flat
    
    train_id_seq = jnp.repeat(jnp.arange(n_dist), n_train)
    test_id_seq = jnp.repeat(jnp.arange(n_dist), n_test)
    
    collection_flat = {
        "train": {
            "X": collection_train[0],
            "y": collection_train[1],
            "id_seq": train_id_seq
        },
        "test": {
            "X": collection_test[0],
            "y": collection_test[1],
            "id_seq": test_id_seq
        }
    }
    
    collection_train, collection_test = collection
    collection_task = {
        "train": {
            "X": collection_train[0],
            "y": collection_train[1]
        },
        "test": {
            "X": collection_test[0],
            "y": collection_test[1],
        },
        "eval": {
            "X": eval_dataset[0],
            "y": eval_dataset[1]
        }
    }
    
    return collection_flat, collection_task


def slice_tasks(datasets, task):
    datasets = deepcopy(datasets)
    train_seq = datasets["train"].pop("id_seq") == task
    test_seq = datasets["test"].pop("id_seq") == task
    
    train = datasets["train"]
    test = datasets["test"]
    
    train = jax.tree_map(lambda x: x[train_seq], train)
    test = jax.tree_map(lambda x: x[test_seq], test)
    
    datasets = {
        "train": train,
        "test": test
    }
    
    return datasets

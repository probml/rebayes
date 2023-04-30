import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr

from rebayes.datasets import classification_data as clf_data
from rebayes.datasets import rotating_mnist_data as rmnist_data


def generate_random_angles(n_tasks, min_angle=0, max_angle=180, key=0):
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    angles = jr.uniform(key, (n_tasks,), minval=min_angle, maxval=max_angle)
    return angles


def rotate_mnist_dataset(X, angles):
    X_rotated = vmap(rmnist_data.rotate_mnist)(X, angles)
    
    return X_rotated


def generate_rotating_mnist_dataset(X, min_angle=0, max_angle=180, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    random_angles = generate_random_angles(len(X), min_angle, max_angle, key)
    X_rotated = rotate_mnist_dataset(X, random_angles)
    
    return X_rotated, random_angles


def generate_rotating_permuted_mnist_regression_dataset(
    n_tasks, ntrain_per_task, nval_per_task, ntest_per_task, min_angle=0, 
    max_angle=180, key=0, fashion=True, mnist_dataset = None
):
    if mnist_dataset is None:
        mnist_dataset = clf_data.load_permuted_mnist_dataset(n_tasks, ntrain_per_task, 
                                                             nval_per_task, ntest_per_task, 
                                                             key, fashion)
    dataset = {
        k: generate_rotating_mnist_dataset(mnist_dataset[k][0], min_angle, max_angle, i)
        for i, k in enumerate(('train', 'val', 'test'))
    }
    
    return dataset

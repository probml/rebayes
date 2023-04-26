import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from rebayes.datasets import classification_data as clf_data
from rebayes.datasets import rotating_mnist_data as rmnist_data


def generate_random_angles(n_tasks, min_angle=0, max_angle=180, key=0):
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    angles = jr.uniform(key, (n_tasks,), minval=min_angle, maxval=max_angle)
    return angles


def generate_rotating_mnist_dataset(X, min_angle=0, max_angle=180, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    X_rotated, y_angles = [], []
    random_angles = generate_random_angles(len(X), min_angle, max_angle, key)
    for i, x in enumerate(X):
        x, angle = np.array(x).reshape((28, 28)), random_angles[i].item()
        x_rotated = rmnist_data.rotate_mnist(x, angle).reshape((1, 28, 28, 1))
        X_rotated.append(jnp.array(x_rotated))
        y_angles.append(angle)
    X_rotated, y_angles = (jnp.array(data) for data in (X_rotated, y_angles))
    
    return X_rotated, y_angles


def generate_rotating_permuted_mnist_regression_dataset(
    n_tasks, ntrain_per_task, nval_per_task, ntest_per_task, min_angle=0, max_angle=180, key=0, fashion=True
):
    pmnist_dataset = clf_data.load_permuted_mnist_dataset(n_tasks, ntrain_per_task, 
                                                          nval_per_task, ntest_per_task, 
                                                          key, fashion)
    dataset = {
        k: generate_rotating_mnist_dataset(pmnist_dataset[k][0], min_angle, max_angle, i)
        for i, k in enumerate(('train', 'val', 'test'))
    }
    
    return dataset

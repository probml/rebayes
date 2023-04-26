from functools import partial

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import tensorflow_datasets as tfds

from rebayes.utils.avalanche import make_avalanche_data


def load_mnist_dataset(fashion=False):
    """Load MNIST train and test datasets into memory."""
    dataset='mnist'
    if fashion:
        dataset='fashion_mnist'
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train[5%:]', batch_size=-1))
    val_ds = tfds.as_numpy(ds_builder.as_dataset(split='train[:5%]', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    
    # Normalize pixel values
    for ds in [train_ds, val_ds, test_ds]:
        ds['image'] = jnp.float32(ds['image']) / 255.
    
    X_train, y_train = (jnp.array(train_ds[key]) for key in ['image', 'label'])
    X_val, y_val = (jnp.array(val_ds[key]) for key in ['image', 'label'])
    X_test, y_test = (jnp.array(test_ds[key]) for key in ['image', 'label'])
    
    dataset = process_dataset(X_train, y_train, X_val, y_val, X_test, y_test, shuffle=True)
        
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
    
    Xval, Yval = (jnp.concatenate(sets) for sets in [Xval_sets, Yval_sets])
    Xte, Yte = (jnp.concatenate(sets) for sets in [Xte_sets, Yte_sets])
    
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


def process_dataset(Xtr, Ytr, Xval, Yval, Xte, Yte, shuffle=False, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
        
    # Reshape data
    Xtr = Xtr.reshape(-1, 1, 28, 28, 1)
    Ytr_ohe = jax.nn.one_hot(Ytr, 10) # one-hot encode labels
    
    # Shuffle data
    if shuffle:
        idx = jr.permutation(key, jnp.arange(len(Xtr)))
        Xtr, Ytr_ohe = Xtr[idx], Ytr_ohe[idx]
    
    dataset = {
        'train': (Xtr, Ytr_ohe),
        'val': (Xval, Yval),
        'test': (Xte, Yte)
    }
    
    return dataset

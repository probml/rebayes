from typing import Sequence
from functools import partial
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import flax.linen as nn
from jax.flatten_util import ravel_pytree
from jax.experimental import host_callback
from jax import jacrev

import torch
from torch.utils.data import TensorDataset, DataLoader

from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM

_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

### MLP

class MLP(nn.Module):
    features: Sequence[int]
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def get_mlp_flattened_params(model_dims, key=0, activation=nn.relu):
    """Generate MLP model, initialize it using dummy input, and
    return the model, its flattened initial parameters, function
    to unflatten parameters, and apply function for the model.
    Args:
        model_dims (List): List of [input_dim, hidden_dim, ..., output_dim]
        key (PRNGKey): Random key. Defaults to 0.
    Returns:
        model: MLP model with given feature dimensions.
        flat_params: Flattened parameters initialized using dummy input.
        unflatten_fn: Function to unflatten parameters.
        apply_fn: fn(flat_params, x) that returns the result of applying the model.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    # Define MLP model
    input_dim, features = model_dims[0], model_dims[1:]
    model = MLP(features, activation)
    dummy_input = jnp.ones((input_dim,))

    # Initialize parameters using dummy input
    params = model.init(key, dummy_input)
    flat_params, unflatten_fn = ravel_pytree(params)

    # Define apply function
    def apply(flat_params, x, model, unflatten_fn):
        return model.apply(unflatten_fn(flat_params), jnp.atleast_1d(x))

    apply_fn = partial(apply, model=model, unflatten_fn=unflatten_fn)

    return model, flat_params, unflatten_fn, apply_fn


### EKF
def initialize_params(flat_params, predict_fn):
    state_dim = flat_params.size
    fcekf_params = ParamsGGSSM(
        initial_mean=flat_params,
        initial_covariance=jnp.eye(state_dim),
        dynamics_function=lambda w, _: w,
        dynamics_covariance = jnp.eye(state_dim) * 1e-4,
        emission_mean_function = lambda w, x: predict_fn(w, x),
        emission_cov_function = lambda w, x: predict_fn(w, x) * (1 - predict_fn(w, x))
    )

    def callback(bel, t, x, y):
        return bel.mean

    return fcekf_params, callback


### SGD

def fit_optax(params, optimizer, input, output, loss_fn, num_epochs, return_history=False):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    if return_history:
        params_history=[]
    
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(input, output)):
            params, opt_state, loss_value = step(params, opt_state, x, y)
            if return_history:
                params_history.append(params)
    
    if return_history:
        return jnp.array(params_history)
    return params

# Generic loss function
def loss_optax(params, x, y, loss_fn, apply_fn):
    y, y_hat = jnp.atleast_1d(y), apply_fn(params, x)
    loss_value = loss_fn(y, y_hat)
    return loss_value.mean()

# Define SGD optimizer
sgd_optimizer = optax.sgd(learning_rate=1e-2)

## Pytorch

def dataloader_to_numpy(dataloader):
  # data = np.array(train_dataloader.dataset) # mangles the shapes
  all_X = []
  all_y = []
  for X, y in dataloader:
    all_X.append(X)
    all_y.append(y)
  X = torch.cat(all_X, dim=0).numpy()
  y = torch.cat(all_y, dim=0).numpy()
  if y.ndim == 1:
      y = y[:, None]
  return X, y

def avalanche_dataloader_to_numpy(dataloader):
  # data = np.array(train_dataloader.dataset) # mangles the shapes
  all_X = []
  all_y = []
  for X, y, t in dataloader: # avalanche uses x,y,t
    all_X.append(X)
    all_y.append(y)
  X = torch.cat(all_X, dim=0).numpy()
  y = torch.cat(all_y, dim=0).numpy()
  if y.ndim == 1:
      y = y[:, None]
  return X, y

def make_avalanche_dataloaders(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch):
    '''Make pytorch dataloaders from avalanche dataset.
    ntrain_per_dist: number of training examples from each distribution (experience).
    batch_size: how many training examples per batch.
    ntest_per_batch: how many test examples per training batch.
    '''
    train_stream = dataset.train_stream
    test_stream = dataset.test_stream
    nexperiences = len(train_stream) # num. distinct distributions
    nbatches_per_dist = int(ntrain_per_dist / ntrain_per_batch)
    ntest_per_dist = ntest_per_batch * nbatches_per_dist
    train_ndx, test_ndx = range(ntrain_per_dist), range(ntest_per_dist)

    train_sets = []
    test_sets = []
    for exp in range(nexperiences):
        ds = train_stream[exp].dataset
        train_set = torch.utils.data.Subset(ds, train_ndx)
        train_sets.append(train_set)

        ds = test_stream[exp].dataset
        test_set = torch.utils.data.Subset(ds, test_ndx)
        test_sets.append(test_set)

    train_set = torch.utils.data.ConcatDataset(train_sets)
    test_set = torch.utils.data.ConcatDataset(test_sets)

    train_dataloader = DataLoader(train_set, batch_size=ntrain_per_batch, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=ntest_per_batch, shuffle=False)
    return train_dataloader, test_dataloader


def make_avalanche_datasets_numpy(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch):
    train_dataloader, test_dataloader = make_avalanche_dataloaders(dataset, ntrain_per_dist, ntrain_per_batch, ntest_per_batch)
    Xtr, Ytr = avalanche_dataloader_to_numpy(train_dataloader)
    Xte, Yte = avalanche_dataloader_to_numpy(test_dataloader)
    #print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape, type(Xtr))
    return Xtr, Ytr, Xte, Yte


def flatten_batch(X):
    if type(X)==torch.Tensor: X = jnp.array(X.numpy())
    sz = jnp.array(list(X.shape))
    batch_size  = sz[0]
    other_size = jnp.prod(sz[1:])
    X = X.flatten().reshape(batch_size, other_size)
    return X

from typing import Sequence
from functools import partial
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit
import flax.linen as nn
from jax.flatten_util import ravel_pytree
from jax.experimental import host_callback
from jax import jacrev
import numpy as np


from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM

# constant is stddev of standard normal truncated to (-2, 2)
TRUNCATED_STD = 20.0 / np.array(.87962566103423978)
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))

### MLP

class MLP(nn.Module):
    features: Sequence[int]
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, x):
        x = x.ravel()
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def scaling_factor(model_dims):
    """This is the factor that is used to scale the
    standardized parameters back into the original space."""
    features = np.array(model_dims)
    biases = features[1:]
    fan_ins = features[:-1]
    num_kernels = features[:-1] * features[1:]
    bias_fanin_kernels = zip(biases, fan_ins, num_kernels)
    factors = []
    for term in bias_fanin_kernels:
        bias, fan_in, num_kernel = (x.item() for x in term)
        factors.extend([1.0] * bias)
        factors.extend([TRUNCATED_STD/np.sqrt(fan_in)] * num_kernel)
    factors = np.array(factors).ravel()

    return factors    

def get_mlp_flattened_params(model_dims, key=0, activation=nn.relu, rescale=False, 
                             zero_ll=False):
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
    if isinstance(input_dim, int):
        dummy_input = jnp.ones((input_dim,))
    else:
        dummy_input = jnp.ones((*input_dim,))
        model_dims = [np.prod(input_dim)] + model_dims[1:]

    # Initialize parameters using dummy input
    params = model.init(key, dummy_input)
    flat_params, unflatten_fn = ravel_pytree(params)
    
    if zero_ll:
        # Zero out the final layer weights
        final_layer_n_params = features[-1] * features[-2] if len(features) > 1 else features[-1]
        flat_params = flat_params.at[-final_layer_n_params:].set(0.0)
    
    scaling = scaling_factor(model_dims, zero_ll) if rescale else 1.0
    flat_params = flat_params / scaling
    rec_fn = lambda x: unflatten_fn(x * scaling)
    
    # Define apply function
    @jit
    def apply_fn(flat_params, x):
        return model.apply(rec_fn(flat_params), jnp.atleast_1d(x))

    return model, flat_params, rec_fn, apply_fn


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

def tree_to_cpu(tree):
    return jax.tree_map(np.array, tree)

def get_subtree(tree, key):
    return jax.tree_map(lambda x: x[key], tree, is_leaf=lambda x: key in x)


def eval_runs(key, num_runs_pc, agent, model, train, test, eval_callback, test_kwargs):
    X_learn, y_learn = train
    _, dim_in = X_learn.shape

    num_devices = jax.device_count()
    num_sims = num_runs_pc * num_devices
    keys = jax.random.split(key, num_sims).reshape(-1, num_devices, 2)
    n_vals = len(X_learn)

    @partial(jax.pmap, in_axes=1)
    @partial(jax.vmap, in_axes=0)
    def evalf(key):
        key_shuffle, key_init = jax.random.split(key)
        ixs_shuffle = jax.random.choice(key_shuffle, n_vals, (n_vals,), replace=False)

        params = model.init(key_init, jnp.ones((1, dim_in)))
        flat_params, _ = ravel_pytree(params)

        bel, output = agent.scan(
            X_learn[ixs_shuffle], y_learn[ixs_shuffle], callback=eval_callback, progress_bar=False, **test_kwargs
        )

        return output


    outputs = evalf(keys)
    outputs = jax.tree_map(lambda x: x.reshape(num_sims, -1), outputs)
    return outputs


def symmetrize_matrix(A):
    """Symmetrize a matrix."""
    return (A + A.T) / 2
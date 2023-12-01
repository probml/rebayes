from functools import partial
from typing import Callable, Sequence

import flax.linen as nn
import jax
from jax import jacrev, jit
from jax.experimental import host_callback
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax


resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode='fan_out', distribution='normal'
)

# .8796... is stddev of standard normal truncated to (-2, 2)
TRUNCATED_STD = 20.0 / np.array(.87962566103423978)
_jacrev_2d = lambda f, x: jnp.atleast_2d(jacrev(f)(x))


class LeNet(nn.Module):
    input_dim: Sequence[int]
    output_dim: int
    activation: nn.Module = nn.relu
    
    @nn.compact
    def __call__(self, x):
        x = x.reshape(self.input_dim)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.output_dim)(x).ravel()
        
        return x
    

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
    

class ResNetBlock(nn.Module):
    output_dim: int
    subsample : bool = False  # If True, we apply a stride inside F
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(
            self.output_dim, kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False
        )(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.activation(z)
        z = nn.Conv(
            self.output_dim, kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False
        )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        if self.subsample:
            x = nn.Conv(
                self.output_dim, kernel_size=(1, 1), strides=(2, 2), 
                kernel_init=resnet_kernel_init
            )(x)
        x_out = self.activation(z + x)
        
        return x_out


class ResNet(nn.Module):
    input_dim: Sequence[int]
    output_dim: int
    hidden_dims: Sequence[int]=(16, 32, 64)
    num_blocks : tuple = (3, 3, 3)
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, x, train=True):
        x = x.reshape(self.input_dim)
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(
            self.hidden_dims[0], kernel_size=(3, 3), 
            kernel_init=resnet_kernel_init, use_bias=False
        )(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.activation(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = ResNetBlock(
                    output_dim=self.hidden_dims[block_idx],
                    activation=self.activation, subsample=subsample
                )(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.output_dim)(x).ravel()
        
        return x



# ------------------------------------------------------------------------------
# Classification

def _initialize_classification(
    key: jnp.ndarray,
    model: nn.Module,
    input_dim: int,
    output_dim: int,
    homogenize_cov: bool = False,
    capture_intermediates: bool = False,
) -> dict:
    """Initialize generic classification model.
    """
    params = model.init(key, jnp.ones(input_dim))['params']
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: \
        model.apply({'params': unflatten_fn(w)}, x,
                    capture_intermediates=capture_intermediates,
                    mutable=['batch_stats'])[0]
    if output_dim == 1:
        # Binary classification
        sigmoid_fn = lambda w, x: \
            jnp.clip(jax.nn.sigmoid(apply_fn(w, x)), 1e-4, 1-1e-4).ravel()
        if capture_intermediates:
            sigmoid_fn = lambda w, x: sigmoid_fn(w, x)[0]
        emission_mean_function = lambda w, x: sigmoid_fn(w, x)
        emission_cov_function = lambda w, x: \
            jnp.atleast_2d(sigmoid_fn(w, x) * (1 - sigmoid_fn(w, x)))
        if homogenize_cov:
            emission_cov_function = lambda w, x: 0.5 * 0.5
    else:
        # Multiclass classification
        emission_mean_function = lambda w, x: jax.nn.softmax(apply_fn(w, x))
        if capture_intermediates:
            emission_mean_function = lambda w, x: \
                emission_mean_function(w, x)[0]
        def emission_cov_function(w, x):
            ps = emission_mean_function(w, x)
            # Add diagonal to avoid singularity
            cov = jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * jnp.eye(len(ps))
            return jnp.atleast_2d(cov)
        if homogenize_cov:
            ps = jnp.ones(output_dim) / output_dim
            emission_cov_function = lambda w, x: \
                jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * jnp.eye(len(ps))
    model_dict = {
        "model": model,
        "flat_params": flat_params,
        "unflatten_fn": unflatten_fn,
        "apply_fn": apply_fn,
        "emission_mean_function": emission_mean_function,
        "emission_cov_function": emission_cov_function,
    }
    
    return model_dict


def initialize_classification_cnn(
    key: int = 0,
    input_dim: Sequence[int] = (1, 28, 28, 1),
    output_dim: int = 10,
    homogenize_cov: bool = False,
    capture_intermediates: bool = False,
    cnn_type="lenet",
) -> dict:
    """Initialize a CNN for classification.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if cnn_type == "lenet":
        model_type = LeNet
    elif cnn_type == "resnet":
        model_type = ResNet
    else:
        raise ValueError(f"Unknown CNN type {cnn_type}")
    model = model_type(input_dim=input_dim, output_dim=output_dim)
    model_dict = _initialize_classification(key, model, input_dim, 
                                            output_dim, homogenize_cov,
                                            capture_intermediates)
    
    return model_dict


def initialize_classification_mlp(
    key: int = 0,
    input_dim: Sequence[int] = (1, 28, 28, 1),
    hidden_dims: Sequence[int] = (500, 500,),
    output_dim: int = 10,
    homogenize_cov: bool = False,
    capture_intermediates: bool = False,
) -> dict:
    """Initialize an MLP for classification.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    features = (*hidden_dims, output_dim)
    model = MLP(features=features)
    model_dict = _initialize_classification(key, model, input_dim,
                                            output_dim, homogenize_cov,
                                            capture_intermediates)
    
    return model_dict


# ------------------------------------------------------------------------------
# Regression

def _initialize_regression(
    key: jnp.ndarray,
    model: nn.Module,
    input_dim: Sequence[int],
    output_dim: int,
    emission_cov: float = 1.0,
    capture_intermediates: bool = False,
) -> dict:
    """Initialize generic regression model.
    """
    params = model.init(key, jnp.ones(input_dim))['params']
    flat_params, unflatten_fn = ravel_pytree(params)
    apply_fn = lambda w, x: \
        model.apply({'params': unflatten_fn(w)}, x,
                     capture_intermediates=capture_intermediates).ravel()
    emission_mean_function = apply_fn
    if capture_intermediates:
        emission_mean_function = lambda w, x: emission_mean_function(w, x)[0]
    emission_cov_function = lambda w, x: emission_cov * jnp.eye(output_dim)
    model_dict = {
        "model": model,
        "flat_params": flat_params,
        "unflatten_fn": unflatten_fn,
        "apply_fn": apply_fn,
        "emission_mean_function": emission_mean_function,
        "emission_cov_function": emission_cov_function,
    }
    
    return model_dict


def initialize_regression_cnn(
    key: int = 0,
    input_dim: Sequence[int] = (1, 28, 28, 1),
    output_dim: int = 1,
    emission_cov: float = 1.0,
    capture_intermediates: bool = False,
) -> dict:
    """Initialize a CNN for regression.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    model = CNN(input_dim=input_dim, output_dim=output_dim)
    model_dict = _initialize_regression(key, model, input_dim, 
                                        output_dim, emission_cov,
                                        capture_intermediates)
    
    return model_dict


def initialize_regression_mlp(
    key: int = 0,
    input_dim: Sequence[int] = (1, 28, 28, 1),
    hidden_dims: Sequence[int] = (500, 500,),
    output_dim: int = 1,
    emission_cov: float = 1.0,
    capture_intermediates: bool = False,
) -> dict:
    """Initialize an MLP for regression.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    features = (*hidden_dims, output_dim)
    model = MLP(features=features)
    model_dict = _initialize_regression(key, model, input_dim,
                                        output_dim, emission_cov,
                                        capture_intermediates)
    
    return model_dict

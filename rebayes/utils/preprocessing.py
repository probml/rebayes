import jax.numpy as jnp

def fourier_basis(domain, b_coef, include_bias=False):
    """
    Example;
    num_basis, input_dim = 7, 1
    b = jax.random.normal(key_basis, (num_basis, input_dim)) * 0.8
    Xf = fourier_basis(X, b); # X.shape == (n_obs, input_dim)
    """
    n_obs = len(domain)
    # We take aj=1
    elements = jnp.einsum("...m,bm->...b", domain, b_coef)
    elements = 2 * jnp.pi * elements
    cos_elements = jnp.cos(elements)
    sin_elements = jnp.sin(elements)
    
    elements = jnp.concatenate([cos_elements, sin_elements], axis=-1)
    if include_bias:
        ones_shape = elements.shape[:-1]
        ones = jnp.ones(ones_shape)[..., None]
        elements = jnp.append(elements, ones, axis=-1)
    return elements

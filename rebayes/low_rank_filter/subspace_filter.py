import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable
from jax.flatten_util import ravel_pytree

def subcify(cls):
    class SubspaceModule(nn.Module):
        dim_in: int
        dim_subspace: int
        init_normal: Callable = nn.initializers.normal()
        init_proj: Callable = nn.initializers.normal()

        def init(self, rngs, *args, **kwargs):
            r1, r2 = jax.random.split(rngs, 2)
            rngs_dict = {"params": r1, "fixed": r2}
            
            return nn.Module.init(self, rngs_dict, *args, **kwargs)

        def setup(self):
            key_dummy = jax.random.PRNGKey(0)
            params = cls().init(key_dummy, jnp.ones((1, self.dim_in)))
            params_all, reconstruct_fn = ravel_pytree(params)
            
            self.dim_full = len(params_all)
            self.reconstruct_fn = reconstruct_fn
            
            self.subspace = self.param(
                "subspace",
                self.init_proj,
                (self.dim_subspace,)
            )

            shape = (self.dim_full, self.dim_subspace)
            init_fn = lambda shape: self.init_proj(self.make_rng("fixed"), shape)
            self.projection = self.variable("fixed", "P", init_fn, shape).value

            shape = (self.dim_full,)
            init_fn = lambda shape: self.init_proj(self.make_rng("fixed"), shape)
            self.bias = self.variable("fixed", "b", init_fn, shape).value

        @nn.compact
        def __call__(self, x):
            params = self.projection @ self.subspace  + self.bias
            params = self.reconstruct_fn(params)
            return cls().apply(params, x)
    
    return SubspaceModule

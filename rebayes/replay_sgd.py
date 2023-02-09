import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial
from rebayes.base import Rebayes
from jaxtyping import Float, Int, Array
from flax.training.train_state import TrainState

class FifoTrainState(TrainState):
    buffer_size: int
    num_obs: int
    buffer_X: Float[Array, "buffer_size dim_features"]
    buffer_y: Float[Array, "buffer_size dim_output"]
    counter: Int[Array, "buffer_size"]

    def _update_buffer(self, step, buffer, item):
        ix_buffer = step % self.buffer_size
        buffer = buffer.at[ix_buffer].set(item)
        return buffer
    
    
    def apply_buffers(self, X, y):
        n_count = self.num_obs
        buffer_X = self._update_buffer(n_count, self.buffer_X, X)
        buffer_y = self._update_buffer(n_count, self.buffer_y, y)
        counter = self._update_buffer(n_count, self.counter, 1.0)
        
        return self.replace(
            num_obs=n_count + 1,
            buffer_X=buffer_X,
            buffer_y=buffer_y,
            counter=counter,
        )
        
        
    @classmethod
    def create(cls, *, apply_fn, params, tx,
               buffer_size, dim_features, dim_output, **kwargs):
        opt_state = tx.init(params)
        buffer_X = jnp.empty((buffer_size, dim_features))
        buffer_y = jnp.empty((buffer_size, dim_output))
        counter = jnp.zeros(buffer_size)
        
        return cls(
            step=0,
            num_obs=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            buffer_size=buffer_size,
            buffer_X=buffer_X,
            buffer_y=buffer_y,
            counter=counter,
            **kwargs
        )


class FSGD(Rebayes):
    """
    FIFO Replay-buffer SGD training procedure
    """
    def __init__(self, lossfn, n_inner=1):
        self.lossfn = lossfn
        self.loss_grad = jax.value_and_grad(self.lossfn, 0)
        self.n_inner = n_inner
    

    def predict_obs(self, bel, X):
        yhat = bel.apply_fn(bel.params, X)
        return yhat

    def predict_state(self, bel):
        return bel
    
    @partial(jax.jit, static_argnums=(0,))
    def _train_step(
        self,
        state: FifoTrainState,
    ) -> Tuple[float, FifoTrainState]:
        """
        """
        X, y = state.buffer_X, state.buffer_y
        loss, grads = self.loss_grad(state.params, state.counter, X, y, state.apply_fn)
        state = state.apply_gradients(grads=grads)
        return loss, state

    @partial(jax.jit, static_argnums=(0,))
    def update_state(self, bel, Xt, yt):
        bel = bel.apply_buffers(Xt, yt) 

        def partial_step(_, bel):
            _, bel = self._train_step(bel)
            return bel
        bel = jax.lax.fori_loop(0, self.n_inner - 1, partial_step, bel)
        # Do not count inner steps as part of the outer step
        _, bel = self._train_step(bel)
        return bel
    
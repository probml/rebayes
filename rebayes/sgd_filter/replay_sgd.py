import jax
import einops
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from functools import partial
from rebayes.base import Rebayes
from jax.flatten_util import ravel_pytree
from jaxtyping import Float, Int, Array
from flax.training.train_state import TrainState

class FifoTrainState(TrainState):
    buffer_size: int
    num_obs: int
    buffer_X: Float[Array, "buffer_size dim_features"]
    buffer_y: Float[Array, "buffer_size dim_output"]
    counter: Int[Array, "buffer_size"]

    @property
    def mean(self):
        return self.params

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
        if isinstance(dim_features, int):   # TODO: Refactor for general case
            buffer_X = jnp.empty((buffer_size, dim_features))
        else:
            buffer_X = jnp.empty((buffer_size, *dim_features))
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


class FifoSGD(Rebayes):
    """
    FIFO Replay-buffer SGD training procedure
    """
    def __init__(self, lossfn, apply_fn=None, init_params=None, tx=None,
                 buffer_size=None, dim_features=None, dim_output=None, n_inner=1):
        self.lossfn = lossfn
        self.apply_fn = apply_fn
        self.params = init_params
        self.tx = tx
        self.buffer_size = buffer_size
        self.dim_features = dim_features
        self.dim_output = dim_output
        self.n_inner = n_inner
        self.loss_grad = jax.value_and_grad(self.lossfn, 0)


    def init_bel(self):
        if self.apply_fn is None:
            raise ValueError("Must provide apply_fn")
        bel_init = FifoTrainState.create(
            apply_fn = self.apply_fn,
            params = self.params,
            tx = self.tx,
            buffer_size = self.buffer_size,
            dim_features = self.dim_features,
            dim_output = self.dim_output
        )
        return bel_init

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


    @partial(jax.jit, static_argnums=(0,4))
    def pred_obs_mc(self, key, bel, x, shape=None):
        """
        Sample observations from the posterior predictive distribution.
        """
        shape = shape or (1,)
        nsamples = np.prod(shape)
        # Belief posterior predictive.
        bel = self.predict_state(bel)
        # TODO: sample from a jax.lax.scan loop over bootstrap of elements in the buffer
        params_sample = jax.tree_map(lambda x: einops.repeat(x, " ... -> b  ...", b=nsamples), bel)  # (b, ...)
        yhat_samples = jax.vmap(self.predict_obs, (0, None))(params_sample, x)
        return yhat_samples


# TODO: replace log_likelihood with TFP distribution
class FifoSGDLaplaceDiag(FifoSGD):
    def __init__(self, lossfn, log_likelihood, apply_fn=None, init_params=None, tx=None,
                 buffer_size=None, dim_features=None, dim_output=None, n_inner=1,
                 prior_precision=1.0):
        super().__init__(lossfn, apply_fn, init_params, tx, buffer_size, dim_features, dim_output, n_inner)
        self.prior_precision = prior_precision
        self.log_likelihood = log_likelihood

    def _get_empirical_fisher(self, bel, X, y):
        """
        Compute the diagonal empirical Fisher information matrix.

        log_likelihood(params, X, y, apply_fn)
        """
        # Gradient of the log-likelihood
        gll = jax.grad(self.log_likelihood, argnums=0)
        # Vectorized gradient of the log-likelihood
        vgll = jax.vmap(gll, (None, 0, 0, None))
        grads = vgll(bel.params, X, y, bel.apply_fn)

        # Empirical Fisher information matrix
        Fdiag = jax.tree_map(lambda x: -(x ** 2).sum(axis=0) - self.prior_precision, grads)
        return Fdiag
    
    def _sample_posterior_params(self, key, bel, nsamples=50):
        """
        Sample parameters from the posterior distribution.
        """
        # Belief posterior predictive.
        mean = self.predict_state(bel).mean
        emp_fisher = self._get_empirical_fisher(bel, bel.buffer_X, bel.buffer_y)
        cov = jax.tree_map(lambda x: -1 / x, emp_fisher)

        mean_flat, unravel_fn = ravel_pytree(mean)
        cov_flat, _ = ravel_pytree(cov)
        nparams = len(mean_flat)

        params_sample = jax.random.normal(key, (nparams, nsamples))
        params_sample = params_sample * jnp.sqrt(cov_flat)[:, None] + mean_flat[:, None]
        params_sample = jax.vmap(unravel_fn, in_axes=-1)(params_sample)
        return params_sample
        
    @partial(jax.jit, static_argnums=(0,4))
    def pred_obs_mc(self, key, bel, x, shape=None):
        """
        Sample observations from the posterior predictive distribution.
        """
        shape = shape or (1,)
        nsamples = np.prod(shape)

        # Sample params
        params_sample = self._sample_posterior_params(key, bel, nsamples=nsamples)
        yhat_samples = jax.vmap(bel.apply_fn, (0, None))(params_sample, x)
        return yhat_samples

    @partial(jax.jit, static_argnames=("self", "n_samples"))
    def nlpd_mc(self, key, bel, x, y, n_samples=30):
        """
        Compute the negative log predictive density (nlpd) as a
        Monte Carlo (MC) estimate.
        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        # 1. Sample params
        params_sample = self._sample_posterior_params(key, bel, nsamples=n_samples)
        # 2. Compute vectorised nlpd (vnlpd)
        vnlpd = jax.vmap(self.log_likelihood, (0, None, None, None))
        vnlpd = jax.vmap(vnlpd, (None, 0, 0, None))
        nlpd_vals = -vnlpd(params_sample, x, y, bel.apply_fn)

        return nlpd_vals.mean(axis=-1)

@partial(jax.jit, static_argnames=("apply_fn",))
def lossfn_rmse(params, counter, X, y, apply_fn):
    """
    Lossfunction for regression problems.
    """
    yhat = apply_fn(params, X).ravel()
    y = y.ravel()
    err = jnp.power(y - yhat, 2)
    loss = (err * counter).sum() / counter.sum()
    return loss


@partial(jax.jit, static_argnames=("apply_fn",))
def lossfn_xentropy(params, counter, X, y, apply_fn):
    """
    Loss function for one-hot encoded classification
    problems.
    """
    yhat = apply_fn(params, X)
    yhat = jax.nn.softmax(yhat).squeeze()
    y = y.squeeze()

    logits = jnp.log(yhat) # B x K
    loss = -jnp.einsum("bk,bk,b->", logits, y, counter) / counter.sum()
    return loss


def init_regression_agent(
    key,
    log_likelihood, # use emission_dist
    model,
    X_init,
    tx,
    buffer_size,
    n_inner=1,
    lossfn=lossfn_rmse,
    prior_precision=1.0,
):
    """
    Initialise regression agent with a given Flax model
    to tackle a 1d-output regression problem.
    """
    apply_fn = model.apply
    init_params = model.init(key, X_init)
    out = model.apply(init_params, X_init)
    dim_output = out.shape[-1]
    dim_features = X_init.shape[-1]

    agent = FifoSGDLaplaceDiag(
        lossfn=lossfn,
        log_likelihood=log_likelihood,
        apply_fn=apply_fn,
        init_params=init_params,
        tx=tx,
        buffer_size=buffer_size,
        dim_output=dim_output,
        dim_features=dim_features,
        n_inner=n_inner,
        prior_precision=prior_precision,
    )
    return agent

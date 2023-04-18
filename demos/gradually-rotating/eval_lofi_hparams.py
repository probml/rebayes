"""
We evaluate the performance of LoFi on the 
rotating MNIST dataset with sinusoidal angle of rotation.
"""

import jax
import flax.linen as nn
import eval_main as ev
from functools import partial
from cfg_regression import get_config

from rebayes.utils.utils import tree_to_cpu

target_digits = [2, 3]
n_classes = len(target_digits)
num_train = 6_000

data = ev.load_data(ev.damp_angle, target_digits, num_train, sort_by_angle=False)
X_train, signal_train, labels_train = data["dataset"]["train"]
X_test, signal_test, labels_test = data["dataset"]["test"]
Y_train = ev.categorise(labels_train)
Y_test = ev.categorise(labels_test)

cfg = get_config()

_, dim_in = X_train.shape
model, tree_params, flat_params, recfn = ev.make_bnn_flax(dim_in, n_classes)
apply_fn = partial(ev.apply_fn_flat, model=model, recfn=recfn)
def emission_mean_fn(w, x): return nn.softmax(apply_fn(w, x))
emission_cov_fn = partial(ev.emission_cov_function, fn_mean=emission_mean_fn)

_, dim_in = data["dataset"]["train"][0].shape

callback_part = partial(ev.callback,
                        apply_fn=emission_mean_fn,
                        X_test=X_train, y_test=Y_train,
                        )

### Lofi---load and train
agent = ev.load_lofi_agent(cfg, flat_params, emission_mean_fn, emission_cov_fn)
callback_lofi = partial(callback_part, recfn=recfn)
bel, outputs_lofi = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_lofi)
bel = jax.block_until_ready(bel)
outputs_lofi = tree_to_cpu(outputs_lofi)

### RSGD---load and train
callback_rsgd = partial(callback_part, recfn=lambda x: x)
apply_fn = partial(ev.apply_fn_unflat, model=model)
agent = ev.load_rsgd_agent(cfg, tree_params, apply_fn, ev.lossfn_fifo, dim_in, n_classes)
bel, outputs_rsgd = agent.scan(X_train, Y_train, progress_bar=True, callback=callback_rsgd)
bel = jax.block_until_ready(bel)
outputs_rsgd = tree_to_cpu(outputs_rsgd)

"""
We evaluate the performance of LoFi on the 
rotating MNIST dataset for classification
with sinusoidal angle of rotation.
"""

import jax
import numpy as np
import flax.linen as nn
import eval_main as ev
from copy import deepcopy
from tqdm.auto import tqdm
from itertools import product
from functools import partial
from cfg_main import get_config

from rebayes.utils.utils import tree_to_cpu

target_digits = [0, 1, 2, 3, 4, 5]
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

callback_lofi = partial(ev.callback,
                        apply_fn=emission_mean_fn,
                        X_test=X_train, y_test=Y_train,
                        recfn=recfn,
                        )

outputs_all = []

# eps = np.array([0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4])
eps = np.array([0, 1e-6, 1e-5, 1e-4])
list_dynamics_weights = 1 - eps
list_dynamics_covariance = eps.copy()

elements = list(product(list_dynamics_weights, list_dynamics_covariance))

for dynamics_weight, dynamics_covariance in tqdm(elements):
    dynamics_weight = float(dynamics_weight)
    dynamics_covariance = float(dynamics_covariance)
    cfg_lofi = deepcopy(cfg)

    cfg_lofi.lofi.dynamics_weight = dynamics_weight
    cfg_lofi.lofi.dynamics_covariance = dynamics_covariance

    agent = ev.load_lofi_agent(cfg_lofi, flat_params, emission_mean_fn, emission_cov_fn)
    bel, output = agent.scan(X_train, Y_train, progress_bar=False, callback=callback_lofi)
    bel = jax.block_until_ready(bel)

    bel = tree_to_cpu(bel)
    output = tree_to_cpu(output)

    res = {
        "config": cfg_lofi,
        "outputs": output,
        "bel": bel,
    }

    outputs_all.append(res)

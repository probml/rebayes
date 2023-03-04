import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import regression_train as benchmark
from typing import Callable
from functools import partial
from rebayes.utils.rotating_mnist_data import load_rotated_mnist
from rebayes.low_rank_filter import lofi
from rebayes.sgd_filter import replay_sgd as rsgd
from jax.flatten_util import ravel_pytree


class MLP(nn.Module):
    n_out: int = 1
    n_hidden: int = 128
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_hidden)(x)
        x = self.activation(x)
        x = nn.Dense(self.n_out)(x)
        return x


def load_data(sort_by_angle: bool = True):
    data = load_rotated_mnist(target_digit=2, sort_by_angle=sort_by_angle)
    train, test = data
    X_train, y_train = train
    X_test, y_test = test

    ymean, ystd = y_train.mean().item(), y_train.std().item()

    y_train = (y_train - ymean) / ystd
    y_test = (y_test - ymean) / ystd

    dataset = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
    }

    res = {
        "dataset": dataset,
        "ymean": ymean,
        "ystd": ystd,
    }

    return res


def train_agents_general(key, model, res, dataset_name, output_path, rank):
    dataset = res["dataset"]
    X_train, _ = dataset["train"]

    # Train, test, warmup
    warmup = dataset["train"]
    warmup = jax.tree_map(lambda x: x[::5], warmup)
    dataset = dataset["train"], dataset["test"], warmup

    ymean = res["ymean"]
    ystd = res["ystd"]
    eval_callback = partial(benchmark.eval_callback_main, ymean=ymean, ystd=ystd)

    _, dim_in = X_train.shape
    params = model.init(key, jnp.ones((1, dim_in)))
    _, reconstruct_fn =  ravel_pytree(params)

    optimizer_eval_kwargs = {
        "init_points": 10,
        "n_iter": 20,
    }

    pbounds = {
        "log_init_cov": (-8, -3),
        "dynamics_weights": (0, 1.0),
        "log_emission_cov": (-7, 0.0),
        "dynamics_log_cov": (-7, 0.0),
    }


    # -------------------------------------------------------------------------
    # Extended Kalman Filter
    # methods = ["fdekf", "vdekf", "fcekf"]
    # methods = ["fdekf", "vdekf"]
    # for method in methods:
    #     res, apply_fn = train_ekf_agent(
    #         params, model, method, dataset, pbounds,
    #         train_callback, eval_callback,
    #         optimizer_eval_kwargs,
    #     )

    #     metric_final = res["output"]["test"][-1]
    #     print(method)
    #     print(f"{metric_final:=0.4f}")
    #     print("-" * 80)
    #     store_results(res, f"{dataset_name}_{method}", output_path)

    # -------------------------------------------------------------------------
    # Low-rank filter

    pbounds_lofi = pbounds.copy()
    pbounds_lofi.pop("dynamics_log_cov")
    # pbounds_lofi["dynamics_covariance"] = None
    # pbounds_lofi["dynamics_covariance"] = (0.0, 1.0)
    pbounds_lofi["dynamics_covariance"] = (0.0, 1e-6) 
    pbounds_lofi["dynamics_weights"] = (0.99, 1.0) 

    methods = ["lofi"]

    params_lofi = lofi.LoFiParams(
        memory_size=rank,
        sv_threshold=0,
        steady_state=True,
        diagonal_covariance=False,
    )
    for method in methods:
        res, apply_fn = benchmark.train_lofi_agent(
            params, params_lofi, model, method, dataset, pbounds_lofi,
            benchmark.train_callback, eval_callback,
            optimizer_eval_kwargs,
        )

        metric_final = res["output"]["test"][-1]
        print(method)
        print(f"{metric_final:=0.4f}")
        print("-" * 80)
        benchmark.store_results(res, f"{dataset_name}_{method}", output_path)

    # -------------------------------------------------------------------------
    # Replay-buffer SGD

    method = "sgd-rb"
    learning_rate = 1e-4
    n_inner = 10

    state_init = rsgd.FifoTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate),
        buffer_size=rank,
        dim_features=dim_in,
        dim_output=1,
    )

    method = "sgd-rb"
    res = benchmark.train_sgd_agent(
        state_init, model, method, dataset, None,
        None, eval_callback, None, n_inner=n_inner,
    )
    metric_final = res["output"]["test"][-1]
    print(method)
    print(f"{metric_final:=0.4f}")
    print("-" * 80)
    benchmark.store_results(res, f"{dataset_name}_{method}", output_path)



if __name__ == "__main__":
    import os
    # output_path = "/home/gerardoduran/documents/rebayes/demos/showdown/output/rotating-mnist"
    # ranks = [1, 5, 10, 15, 20, 30, 40, 50]
    ranks = [1, 2]
    for rank in ranks:
        print(f"------------------ Rank: {rank} ------------------")
        dataset_name = f"sorted-rotating-mnist-2-mlp-rank{rank}"
        output_path = os.environ.get("REBAYES_OUTPUT")
        if output_path is None:
            output_path = "/tmp/rebayes"
            os.mkdir(output_path)
        print(f"Dataset name: {dataset_name}")
        print(f"Output path: {output_path}")

        data = load_data()
        model = MLP(n_out=1, n_hidden=100)
        key = jax.random.PRNGKey(314)

        train_agents_general(key, model, data, dataset_name, output_path, rank)

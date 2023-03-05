import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import regression_train as benchmark
from typing import Callable
from functools import partial
from rebayes.utils.rotating_mnist_data import load_rotated_mnist
from rebayes.low_rank_filter import lofi
from rebayes import base
from rebayes.sgd_filter import replay_sgd as rsgd
from jax.flatten_util import ravel_pytree


class MLP(nn.Module):
    n_out: int = 1
    n_hidden: int = 100
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
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
    X_train, y_train = dataset["train"]
    X_test, y_test = dataset["test"]

    # Train, test, warmup
    warmup = dataset["train"]
    warmup = jax.tree_map(lambda x: x[::5], warmup)
    # dataset = dataset["train"], dataset["test"], warmup
    dataset = warmup, dataset["test"], warmup

    ymean = res["ymean"]
    ystd = res["ystd"]
    eval_callback = partial(benchmark.eval_callback_main, ymean=ymean, ystd=ystd)

    _, dim_in = X_train.shape
    params = model.init(key, jnp.ones((1, dim_in)))
    params_flat, reconstruct_fn =  ravel_pytree(params)

    optimizer_eval_kwargs = {
        "init_points": 10,
        "n_iter": 30,
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
    #     res, apply_fn = benchmark.train_ekf_agent(
    #         params, model, method, dataset, pbounds,
    #         benchmark.train_callback, eval_callback,
    #         optimizer_eval_kwargs,
    #     )

    #     metric_final = res["output"]["test"][-1]
    #     print(method)
    #     print(f"{metric_final:=0.4f}")
    #     print("-" * 80)
    #     benchmark.store_results(res, f"{dataset_name}_{method}", output_path)

    # -------------------------------------------------------------------------
    # Low-rank filter

    pbounds_lofi = pbounds.copy()
    pbounds_lofi.pop("dynamics_log_cov")
    # pbounds_lofi["dynamics_covariance"] = None
    pbounds_lofi["dynamics_covariance"] = (0.0, 1.0)

    methods = ["lofi", "lofi_orth"]

    params_lofi = lofi.LoFiParams(
        memory_size=rank,
        sv_threshold=0,
        steady_state=False,
        diagonal_covariance=False,
    )
    for method in methods:
        res, apply_fn, hparams = benchmark.train_lofi_agent(
            params, params_lofi, model, method, dataset, pbounds_lofi,
            benchmark.train_callback, eval_callback,
            optimizer_eval_kwargs,
        )

        metric_final = res["output"]["test"][-1]
        print(method)
        print(f"{metric_final:=0.4f}")
        print("-" * 80)
        benchmark.store_results(res, f"{dataset_name}_{method}", output_path)
    
    # Generalised LoFi
    method = "lofi"
    params_lofi = lofi.LoFiParams(
        memory_size=rank,
        sv_threshold=0,
        steady_state=False,
        diagonal_covariance=True,
    )

    res, apply_fn, hparams = benchmark.train_lofi_agent(
        params, params_lofi, model, method, dataset, pbounds_lofi,
        benchmark.train_callback, eval_callback,
        optimizer_eval_kwargs,
    )
    method = "lofi_diag"
    res["method"] = method

    metric_final = res["output"]["test"][-1]
    print(method)
    print(f"{metric_final:=0.4f}")
    print("-" * 80)
    benchmark.store_results(res, f"{dataset_name}_{method}", output_path)

    # -------------------------------------------------------------------------
    # Replay-buffer SGD

    pbounds_rsgd = {
        "learning_rate": (1e-6, 1e-2),
        "n_inner": (1, 100),
    }

    method = "sgd-rb"
    res = benchmark.train_sgd_agent(
        params, model, method, dataset, pbounds_rsgd,
        benchmark.train_callback, eval_callback,
        optimizer_eval_kwargs, rank=rank,
    )
    metric_final = res["output"]["test"][-1]
    print(method)
    print(f"{metric_final:=0.4f}")
    print("-" * 80)
    benchmark.store_results(res, f"{dataset_name}_{method}", output_path)

    # -------------------------------------------------------------------------
    # ORFIT
    method = "orfit"
    params_orfit = base.RebayesParams(
        initial_mean=params_flat,
        emission_mean_function=apply_fn,
        **hparams
    )

    X_train, y_train = warmup
    agent = lofi.RebayesLoFi(params_orfit, params_lofi, method=method)
    test_kwargs = {"X_test": X_test, "y_test": y_test, "apply_fn": apply_fn}
    _, output = agent.scan(
        X_train, y_train, callback=eval_callback, progress_bar=False, **test_kwargs
    )
    res = {
        "output": output,
        "method": method,
    }
    metric_final = res["output"]["test"][-1]
    print(method)
    print(f"{metric_final:=0.4f}")
    print("-" * 80)
    benchmark.store_results(res, f"{dataset_name}_{method}", output_path)


if __name__ == "__main__":
    import os
    # output_path = "/home/gerardoduran/documents/rebayes/demos/showdown/output/rotating-mnist"
    ranks = [2, 5, 10, 15, 20, 30, 40, 50]
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        output_path = "/tmp/rebayes"
        os.mkdir(output_path)
    print(f"Output path: {output_path}")

    for rank in ranks:
        print(f"------------------ Rank: {rank} ------------------")
        dataset_name = f"sorted-rotating-mnist-2-mlp-rank{rank:02}"
        print(f"Dataset name: {dataset_name}")

        data = load_data()
        model = MLP(n_out=1, n_hidden=100)
        key = jax.random.PRNGKey(314)

        train_agents_general(key, model, data, dataset_name, output_path, rank)

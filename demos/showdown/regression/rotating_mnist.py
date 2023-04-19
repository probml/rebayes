import jax
import jax.numpy as jnp
import flax.linen as nn
import regression_train as benchmark
from typing import Callable
from functools import partial
from rebayes.datasets.rotating_mnist_data import load_rotated_mnist
from rebayes.low_rank_filter import lofi
from rebayes import base
from rebayes.sgd_filter import replay_sgd as rsgd
from jax.flatten_util import ravel_pytree


class CNN(nn.Module):
    n_out: int = 1
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, 28, 28, 1))
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = self.activation(x)
        x = nn.Dense(features=self.n_out)(x)
        x = x.squeeze(-1)
        return x


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


def train_xkf_agents(params, dataset, pbounds, eval_callback, method, optimizer_eval_kwargs):
    res, apply_fn = benchmark.train_ekf_agent(
        params, model, method, dataset, pbounds,
        benchmark.train_callback, eval_callback,
        optimizer_eval_kwargs,
    )

    metric_final = res["output"]["test"][-1]
    print(method)
    print(f"{metric_final:=0.4f}")
    print("-" * 80)
    benchmark.store_results(res, f"{dataset_name}_{method}", output_path)
    return True


def train_agents_general(key, model, res, dataset_name, output_path, rank):
    dataset = res["dataset"]
    X_test, y_test = dataset["test"]
    # Use train as warmup
    X_train, y_train = dataset["train"]

    # Train, test, warmup
    dataset = dataset["train"], dataset["test"], dataset["train"][:200]

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
        "log_init_cov": (-10, 0),
        "log_dynamics_weights": (-5, 0), # Change to log-space close to 1.0
        "log_emission_cov": (-80, 0.0),
        "dynamics_log_cov": (-80, 0.0),
    }

    # -------------------------------------------------------------------------
    # Extended Kalman Filter
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
    # pbounds_lofi.pop("dynamics_log_cov")
    # pbounds_lofi["dynamics_covariance"] = None # If steady-state
    # pbounds_lofi["dynamics_covariance"] = (0.0, 1.0)
    pbounds_lofi["log_inflation"] = (-40, 0.0)

    method = "lofi"
    params_lofi = lofi.LoFiParams(
        memory_size=rank,
        steady_state=False,
    )

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

    # -------------------------------------------------------------------------
    # Replay-buffer SGD
    pbounds_rsgd = {
        "learning_rate": (1e-6, 1e-2),
        "n_inner": (1, 1),
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
    # Low-rank Variational Gaussian Approximation (LRVGA)

    # random_state = 314
    # optimizer_eval_kwargs = {
    #     "init_points": 5,
    #     "n_iter": 0,
    # }

    # pbounds_lrvga = {
    #     "std": (-0.34, 0.0),
    #     "sigma2": (-4, 0.0),
    #     "eps": (-10, -4),
    # }

    # n_outer = 2
    # n_inner = 3
    # fwd_link = partial(benchmark.fwd_link_main, model=model, reconstruct_fn=reconstruct_fn)
    # res = benchmark.train_lrvga_agent(
    #     key, apply_fn, model, dataset, dim_rank=rank, n_inner=n_inner, n_outer=n_outer,
    #     pbounds=pbounds_lrvga, eval_callback=eval_callback,  fwd_link=fwd_link,
    #     optimizer_eval_kwargs=optimizer_eval_kwargs,
    #     random_state=random_state,
    # ) 
    # method = res["method"]

    # metric_final = res["output"]["test"][-1]
    # print(method)
    # print(f"{metric_final:=0.4f}")
    # print("-" * 80)
    # benchmark.store_results(res, f"{dataset_name}_{method}", output_path)


if __name__ == "__main__":
    import os
    # output_path = "/home/gerardoduran/documents/rebayes/demos/showdown/output/rotating-mnist"
    ranks = [5, 10, 15, 20, 50]
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        output_path = "/tmp/rebayes"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    print(f"Output path: {output_path}")

    for rank in ranks:
        print(f"------------------ Rank: {rank} ------------------")
        dataset_name = f"rotating-mnist-2-mlp-rank{rank:02}"
        print(f"Dataset name: {dataset_name}")

        data = load_data(sort_by_angle=True)
        # model = MLP(n_out=1, n_hidden=100)
        model = CNN(n_out=1)
        key = jax.random.PRNGKey(314)
        train_agents_general(key, model, data, dataset_name, output_path, rank)

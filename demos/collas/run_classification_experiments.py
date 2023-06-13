import argparse
from functools import partial
import json
import os
from typing import Callable
from pathlib import Path

import flax.linen as nn
import jax.random as jr
import optax

import demos.collas.datasets.mnist_data as mnist_data
import rebayes.utils.models as models
import rebayes.utils.callbacks as callbacks
import demos.collas.classification.clf_hparam_tune as hparam_tune

AGENT_TYPES = ["lofi", "fdekf", "vdekf", "sgd-rb", "adam-rb"]


def _check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    
    return ivalue


def _process_agent_args(agent_args, ranks, output_dim, problem):
    agents = {}
    sgd_loss_fn = optax.softmax_cross_entropy if output_dim >= 2 \
        else optax.sigmoid_binary_cross_entropy
    
    # Bounds for tuning
    sgd_pbounds = {
        "log_learning_rate": (-10.0, 0.0),
    }
    if problem == "stationary":
        filter_pbounds = {
            'log_init_cov': (-10, 0.0),
            'log_dynamics_weights': (-90, -90),
            'log_dynamics_cov': (-90, -90),
            'log_alpha': (-90, -90),
        }
    else:
        filter_pbounds = {
            'log_init_cov': (-10, 0),
            'log_dynamics_weights': (-30, 0),
            'log_dynamics_cov': (-30, 0),
            'log_alpha': (-30, 0),
        }
    
    # Create agents
    if "lofi" in agent_args:
        agents.update({
            f'lofi-{rank}': {
                'memory_size': rank,
                'inflation': "hybrid",
                'lofi_method': "diagonal",
                'pbounds': filter_pbounds,
            } for rank in ranks
        })
    if "fdekf" in agent_args:
        agents["fdekf"] = {'pbounds': filter_pbounds}
    if "vdekf" in agent_args:
        agents["vdekf"] = {'pbounds': filter_pbounds}
    if "sgd-rb" in agent_args:
        agents.update({
            f'sgd-rb-{rank}': {
                'loss_fn': sgd_loss_fn,
                'buffer_size': rank,
                'dim_output': output_dim,
                "optimizer": "sgd",
                'pbounds': sgd_pbounds,
            } for rank in ranks
        })
    if "adam-rb" in agent_args:
        agents.update({
            f'adam-rb-{rank}': {
                'loss_fn': sgd_loss_fn,
                'buffer_size': rank,
                'dim_output': output_dim,
                "optimizer": "adam",
                'pbounds': sgd_pbounds,
            } for rank in ranks
        })
    
    return agents


def tune_and_store_hyperparameters(
    model_init_fn: Callable,
    dataset: dict,
    agents: dict,
    hparam_path: Path,
    problem: str,
) -> dict:
    """Tune and store hyperparameters.

    Args:
        model_dict (dict): Dictionary of model parameters.
        dataset (dict): Dictionary of dataset parameters.
        agents (dict): Dictionary of agent parameters.
        hparam_path (Path): Path to hyperparmeter directory.
        problem (str): Type of MNIST classification task.

    Returns:
        hparams (dict): Dictionary of tuned hyperparameters.
    """
    hparam_path.mkdir(parents=True, exist_ok=True)
    
    # Set up tuning objective
    ll_softmax = lambda logits, labels: \
        -optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    ll_softmax_eval_fn = \
        partial(callbacks.clf_evaluate_function, loss_fn=ll_softmax)
    ll_callback = \
        partial(callbacks.cb_clf_eval, evaluate_fn=ll_softmax_eval_fn)
    
    hparams = {}
    for agent_name, agent_params in agents.items():
        print(f"Tuning {agent_name}...")
        pbounds = agent_params.pop("pbounds")
        optimizer = hparam_tune.create_optimizer(
            model_init_fn, pbounds, dataset["train"], dataset["val"],
            ll_callback, agent_name, verbose=2, callback_at_end=False,
            **agent_params
        )
        optimizer.maximize(init_points=1, n_iter=1)
        best_hparams = hparam_tune.get_best_params(optimizer, agent_name)
        # Store as json
        with open(Path(hparam_path, f"{agent_name}.json"), "w") as f:
            json.dump(best_hparams, f)
        hparams[agent_name] = best_hparams

    return hparams


def main(cl_args):
    # Set output path
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        output_path = Path("classification", "output", cl_args.problem)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Set config path
    config_path = os.environ.get("REBAYES_CONFIG")
    if config_path is None:
        config_path = Path("classification", "configs")
        config_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_load_fn = mnist_data.Datasets[cl_args.problem+"-mnist"]
    dataset = dataset_load_fn(fashion=cl_args.dataset=="f-mnist")
    
    # Initialize model
    if cl_args.model == "cnn":
        model_init_fn = models.initialize_classification_cnn
    else: # cl_args.model == "mlp"
        model_init_fn = models.initialize_classification_mlp
    model_dict = model_init_fn()
    
    # Set up agents
    output_dim = 2 if cl_args.problem == "split" else 10
    agents = _process_agent_args(cl_args.agents, cl_args.ranks, output_dim,
                                 cl_args.problem)
    
    # Set up hyperparameter tuning
    hparam_path = Path(config_path, cl_args.problem, 
                       cl_args.dataset, cl_args.model)
    if cl_args.tune:
        hparams = tune_and_store_hyperparameters(model_init_fn, dataset, agents,
                                                 hparam_path, cl_args.problem)
    else:
        hparams = {}
        for agent_name in agents:
            # Check if hyperparameters are specified in config file
            agent_hparam_path = Path(hparam_path, agent_name+".json")
            try:
                # Load json file
                with open(agent_hparam_path, "r") as f:
                    hparams[agent_name] = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Hyperparameter {agent_hparam_path} "
                                        "not found.")
    print(hparams)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Problem type (stationary, permuted, rotated, or split)
    parser.add_argument("--problem", type=str, default="stationary",
                        choices=["stationary", "permuted", "rotated", "split"])
    
    # Type of dataset (mnist or f-mnist)
    parser.add_argument("--dataset", type=str, default="mnist", 
                        choices=["mnist", "f-mnist"])
    
    # Type of model (mlp or cnn)
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "cnn"])
    
    # Tune the hyperparameters of the agents
    parser.add_argument("--tune", action="store_true")
    
    # List of ranks to use for the agents
    parser.add_argument("--ranks", type=_check_positive_int, nargs="+",
                        default=[1, 10,])
    
    # List of agents to use
    parser.add_argument("--agents", type=str, nargs="+", default=AGENT_TYPES,
                        choices=AGENT_TYPES)
    
    args = parser.parse_args()
    main(args)
    
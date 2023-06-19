import argparse
from functools import partial
import json
import os
from typing import Callable
from pathlib import Path
import pickle

import jax.random as jr
from jax.tree_util import tree_map

import demos.collas.datasets.mnist_data as mnist_data
import rebayes.utils.models as models
import rebayes.utils.callbacks as callbacks
import demos.collas.hparam_tune as hparam_tune
import demos.collas.classification.clf_train as train_utils

AGENT_TYPES = ["lofi", "fdekf", "vdekf", "sgd-rb", "adam-rb"]


def _check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    
    return ivalue


def _check_positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float")
    
    return fvalue


def _process_angle_fn(problem):
    match problem:
        case "iid":
            angle_fn = mnist_data.generate_random_angles
        case "amplified":
            angle_fn = mnist_data.generate_amplified_angles
        case "random_walk":
            angle_fn = mnist_data.generate_random_walk_angles
        case other:
            # TODO: Implement permuted rotating regression
            angle_fn = None
    
    return angle_fn


def _process_agent_args(agent_args, ranks, output_dim, problem, obs_scale):
    agents = {}
    sgd_loss_fn = partial(callbacks.nll_reg, scale=obs_scale)
    
    # Bounds for tuning
    sgd_pbounds = {
        "log_init_cov": (-10.0, 0.0),
        "log_learning_rate": (-10.0, 0.0),
    }
    if problem == "iid":
        filter_pbounds = {
            'log_init_cov': (-10, 0.0),
            'log_1m_dynamics_weights': (-90, -90),
            'log_dynamics_cov': (-90, -90),
            'log_alpha': (-30, 0),
        }
    else:
        filter_pbounds = {
            'log_init_cov': (-10, 0),
            'log_1m_dynamics_weights': (-30, 0),
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


def _eval_metric(
    obs_scale: float,
    problem: str,
    nll_method: str,
) -> dict:
    """Get evaluation metric for classification problem type.
    """
    if problem == "iid":
        if nll_method == "nll":
            result = {
                "val": partial(
                    callbacks.cb_eval,
                    evaluate_fn=callbacks.generate_ll_reg_eval_fn(obs_scale)
                ),
                "test": partial(
                    callbacks.cb_eval,
                    evaluate_fn=partial(callbacks.reg_eval_fn, scale=obs_scale)
                )
            }
        else: # nlpd-mc
            result = {
                "val": lambda *args, **kwargs: tree_map(
                    lambda x: -x, callbacks.cb_reg_nlpd_mc(*args, **kwargs)
                ),
                "test": callbacks.cb_reg_nlpd_mc,
            }
    else: # Nonstationary
        result = {
            "val": partial(callbacks.cb_osa,
                            evaluate_fn=partial(callbacks.ll_reg,
                                                scale=obs_scale),
                            label="log_likelihood"),
            "test": partial(callbacks.cb_reg_sup,
                            ymean=0.0, ystd=1.0, 
                            only_window_eval=True)
        }
    
    return result


def tune_and_store_hyperparameters(
    hparam_path: Path,
    model_init_fn: Callable,
    dataset_load_fn: Callable,
    agents: dict,
    val_callback: Callable,
    verbose: int = 2,
    n_explore: int = 20,
    n_exploit: int = 25,
) -> dict:
    """Tune and store hyperparameters.

    Args:
        hparam_path (Path): Path to hyperparmeter directory.
        model_init_fn (Callable): Model initialization function.
        dataset_load_fn (Callable): Dataset loading function.
        agents (dict): Dictionary of agent parameters.
        val_callback (Callable): Tuning objective.
        verbosity (int, optional): Verbosity level for Bayesian optimization.
        n_explore (int, optional): Number of random exploration steps
            for Bayesian optimization. Defaults to 20.
        n_exploit (int, optional): Number of exploitation steps for
            Bayesian optimization. Defaults to 25.

    Returns:
        hparams (dict): Dictionary of tuned hyperparameters.
    """
    hparam_path.mkdir(parents=True, exist_ok=True)
    dataset = dataset_load_fn()
    
    hparams = {}
    for agent_name, agent_params in agents.items():
        print(f"Tuning {agent_name}...")
        pbounds = agent_params.pop("pbounds")
        optimizer = hparam_tune.create_optimizer(
            model_init_fn, pbounds, dataset["train"], dataset["val"],
            val_callback, agent_name, verbose=verbose, callback_at_end=False,
            classification=False, **agent_params
        )
        optimizer.maximize(init_points=n_explore, n_iter=n_exploit)
        best_hparams = hparam_tune.get_best_params(optimizer, agent_name,
                                                   classification=False)
        # Store as json
        with open(Path(hparam_path, f"{agent_name}.json"), "w") as f:
            json.dump(best_hparams, f)
        hparams[agent_name] = best_hparams

    return hparams


def evaluate_and_store_result(
    output_path: Path,
    model_init_fn: Callable,
    dataset_load_fn: Callable,
    optimizer_dict: dict,
    eval_callback: Callable,
    agent_name: str,
    problem: str,
    n_iter: int=20,
    key: int=0,
    **kwargs: dict,
) -> dict:
    """Evaluate and store results.

    Args:

        model_init_fn (Callable): Model initialization function.
        dataset_load_fn (Callable): Dataset loading function.
        optimizer_dict (dict): Dictionary of optimizer parameters.
        eval_callback (Callable): Evaluation callback.
        problem (str): Problem type.
        n_iter (int, optional): Number of random initializations. Defaults to 20.
        key (int, optional): Random seed. Defaults to 0.

    Returns:
        result (dict): Dictionary of results.
    """
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if problem == "iid" or problem == "amplified" or problem == "random-walk":
        eval_fn = train_utils.eval_agent_stationary
    else:
        eval_fn = train_utils.eval_agent_nonstationary
    result = eval_fn(model_init_fn, dataset_load_fn, optimizer_dict,
                     eval_callback, n_iter, key, **kwargs)
    
    # Store result
    with open(Path(output_path, f"{agent_name}.pkl"), "wb") as f:
        pickle.dump(result, f)
    
    return result


def main(cl_args):
    # Set output path
    output_path = os.environ.get("REBAYES_OUTPUT")
    problem_name = cl_args.problem + "-" + str(cl_args.ntrain)
    if output_path is None:
        output_path = Path("regression", "outputs", problem_name,
                           cl_args.dataset, cl_args.model, cl_args.nll_method)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Set config path
    config_path = os.environ.get("REBAYES_CONFIG")
    if config_path is None:
        config_path = Path("regression", "configs")
    Path(config_path).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    angle_fn = _process_angle_fn(cl_args.problem)
    dataset = mnist_data.generate_rmnist_experiment(cl_args.ntrain, angle_fn)
    dataset_load_fn, kwargs = dataset.values()
    ntrain = None
    if kwargs is not None and "ntrain" in kwargs:
        ntrain = kwargs["ntrain"]
    base_dataset = mnist_data.load_target_digit_dataset(
        fashion=cl_args.dataset=="f-mnist",
        target_digit=cl_args.target_digit, n=ntrain,
    )
    dataset_load_fn = partial(dataset_load_fn, dataset=base_dataset)
    eval_metric = _eval_metric(cl_args.obs_scale, cl_args.problem, 
                               cl_args.nll_method)
    
    # Initialize model
    if cl_args.model == "cnn":
        model_init_fn = partial(models.initialize_regression_cnn, 
                                emission_cov=cl_args.obs_scale**2)
    else: # cl_args.model == "mlp"
        model_init_fn = partial(models.initialize_regression_mlp, 
                                emission_cov=cl_args.obs_scale**2)
    
    # Set up agents
    output_dim = 1
    agents = _process_agent_args(cl_args.agents, cl_args.ranks, output_dim,
                                 cl_args.problem, cl_args.obs_scale)
    
    # Set up hyperparameter tuning
    hparam_path = Path(config_path, problem_name, cl_args.dataset, 
                       cl_args.model, cl_args.nll_method)
    if cl_args.tune:
        agent_hparams = \
            tune_and_store_hyperparameters(hparam_path, model_init_fn, 
                                           dataset_load_fn, agents,
                                           eval_metric["val"], cl_args.verbose, 
                                           cl_args.n_explore, cl_args.n_exploit)
    else:
        agent_hparams = {}
        for agent_name in agents:
            # Check if hyperparameters are specified in config file
            agent_hparam_path = Path(hparam_path, agent_name+".json")
            try:
                # Load json file
                with open(agent_hparam_path, "r") as f:
                    agent_hparams[agent_name] = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Hyperparameter {agent_hparam_path} "
                                        "not found.")
    
    # Evaluate agents
    kwargs["scale"] = cl_args.obs_scale
    for agent_name, hparams in agent_hparams.items():
        print(f"Evaluating {agent_name}...")
        agent_kwargs = agents[agent_name]
        if "pbounds" in agent_kwargs:
            agent_kwargs.pop("pbounds")
        optimizer_dict = hparam_tune.build_estimator(model_init_fn, hparams,
                                                     agent_name, 
                                                     classification=False,
                                                     **agent_kwargs)
        _ = evaluate_and_store_result(output_path, model_init_fn,
                                      dataset_load_fn, optimizer_dict,
                                      eval_metric["test"], agent_name,
                                      cl_args.problem, cl_args.n_iter,
                                      **kwargs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Problem type (stationary, permuted, rotated, or split)
    parser.add_argument("--problem", type=str, default="iid",
                        choices=["iid", "amplified", "random-walk", "permuted"])
    
    # Type of dataset (mnist or f-mnist)
    parser.add_argument("--dataset", type=str, default="f-mnist", 
                        choices=["mnist", "f-mnist"])
    
    # Target digit
    parser.add_argument("--target_digit", type=int, default=2,
                        choices=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Number of training samples
    parser.add_argument("--ntrain", type=_check_positive_int, default=2_000)
    
    # Type of model (mlp or cnn)
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "cnn"])
    
    # Observation noise
    parser.add_argument("--obs_scale", type=_check_positive_float, default=15.0)
    
    # Negative log likelihood evaluation method
    parser.add_argument("--nll_method", type=str, default="nll", 
                        choices=["nll", "nlpd-mc"])
    
    # Tune the hyperparameters of the agents
    parser.add_argument("--tune", action="store_true")
    
    # Set the number of exploration steps
    parser.add_argument("--n_explore", type=int, default=10)
    
    # Set the number of exploitation steps
    parser.add_argument("--n_exploit", type=int, default=15)
    
    # Set the verbosity of the Bayesopt procedure
    parser.add_argument("--verbose", type=int, default=2,
                        choices=[0, 1, 2])
    
    # List of ranks to use for the agents
    parser.add_argument("--ranks", type=_check_positive_int, nargs="+",
                        default=[1, 10,])
    
    # List of agents to use
    parser.add_argument("--agents", type=str, nargs="+", default=AGENT_TYPES,
                        choices=AGENT_TYPES)
    
    # Number of random initializations for evaluation
    parser.add_argument("--n_iter", type=int, default=100)
    
    args = parser.parse_args()
    main(args)
    
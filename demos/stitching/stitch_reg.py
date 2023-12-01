import argparse
from functools import partial
import json
from pathlib import Path
import pickle

from bayes_opt import BayesianOptimization
import flax
from flax.training.train_state import TrainState
from jax import vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
import optax
import seaborn as sns
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

import rebayes.datasets.datasets as dataset
import demos.collas.datasets.dataloaders as dataloaders
import rebayes.utils.models as models
import rebayes.utils.callbacks as callbacks
import rebayes.sgd_filter.sgd as sgd
import demos.collas.hparam_tune as hparam_tune
import demos.collas.train_utils as train_utils
import demos.collas.run_classification_experiments as experiments

AGENT_TYPES = ["fcekf", "neural-linear", "full-neural-linear",
               "neural-stitching"]

PBOUNDS = {
    "fcekf": {
        'log_init_cov': (-10.0, 2.0),
        'log_1m_dynamics_weights': (-90, -90),
        'log_dynamics_cov': (-90, -90),
        'log_alpha': (-90, -90),
    },
    "sgd": {
        'log_learning_rate': (-10.0, 2.0)
    }
}


def _check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    
    return ivalue


def _f1(x, key, obs_var):
    w = jnp.array([-0.6667, -0.6012, -1.0172, -0.7687, 1.4680, -0.1678])
    fx = w @ jnp.power(x, jnp.arange(len(w)))
    fx *= jnp.sin(jnp.pi * x)
    fx *= jnp.exp(-0.5 * (x**2)) / jnp.sqrt(2 * jnp.pi)
    
    return fx + obs_var * jr.normal(key)


def _f2(x, key, obs_var):
    fx = jnp.exp(jnp.abs(x))
    fx *= jnp.cos(2*jnp.pi * x)
    
    return fx + obs_var * jr.normal(key)


def generate_dataset(n_train, obs_var, gen_fn=_f1, key=0, in_between=False):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    
    if in_between:
        mask = jr.bernoulli(key, p=0.5, shape=(n_train,))
        key1, key2, subkey = jr.split(subkey, 3)
        X_train = (-2 * jr.uniform(key1, (n_train,)) - 1) * mask + \
            (2 * jr.uniform(key2, (n_train,)) + 1) * (1. - mask)
    else:
        X_train = jr.normal(key, (n_train,))
    
    keys = jr.split(subkey, n_train)
    y_train = vmap(gen_fn, (0, 0, None))(X_train, keys, obs_var)

    return X_train, y_train


def generate_sub_networks(model_init_fn, flat_params, layer_start=0, layer_end=-1):
    params_dict = model_init_fn(0)["unflatten_fn"](flat_params)
    model = model_init_fn(0)["model"]
    features, activation = model.features, model.activation
    if layer_end == -1 or layer_end >= len(features):
        layer_end = len(features)
        activation = lambda x: x
    if layer_start > len(features):
        raise ValueError(f"layer_start must be less than {len(features)}")
    if layer_end < layer_start:
        raise ValueError(f"layer_end must be greater than layer_start")
    model.features = features[layer_start:layer_end]
    if layer_start == 0:
        input_dim, *_ = params_dict["Dense_0"]["kernel"].shape
    else:
        input_dim = features[layer_start - 1]
    params = {
        f"Dense_{i-layer_start}": 
            params_dict[f"Dense_{i}"] for i in range(layer_start, layer_end)
    }
    params = flax.core.frozen_dict.freeze(params)
    apply_fn = lambda x: activation(model.apply({"params": params}, x))
    
    sub_model_dict = {
        "model": model,
        "params": params,
        "apply_fn": apply_fn,
        "activation": activation,
        "input_dim": input_dim,
        "output_dim": features[layer_end - 1],
    }
    
    return sub_model_dict


def construct_linear_stitching(model1, model2, emission_cov=0.3, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    input_dim, output_dim = model1["output_dim"], model2["input_dim"]
    
    # Construct linear stitching
    linear_model = models.MLP(features=[output_dim,], activation=lambda x: x)
    params = linear_model.init(key, jnp.ones((input_dim,)))["params"]
    flat_params, unflatten_fn = ravel_pytree(params)
    linear_apply_fn = lambda w, x: linear_model.apply({"params": unflatten_fn(w)}, x)
    
    def apply_fn(w, x):
        input = model1["apply_fn"](x)
        linear_output = linear_apply_fn(w, input)
        output = model2["apply_fn"](linear_output).ravel()
        return output
    
    emission_mean_function = apply_fn
    emission_cov_function = lambda w, x: emission_cov * jnp.eye(model2["output_dim"])
    
    linear_model_dict = {
        "model": linear_model,
        "flat_params": flat_params,
        "unflatten_fn": unflatten_fn,
        "apply_fn": apply_fn,
        "emission_mean_function": emission_mean_function,
        "emission_cov_function": emission_cov_function,
    }
    
    return linear_model_dict


def train_offline_sgd(
    model_init_fn, X, y, batch_size=8, num_epochs=1_000, learning_rate=1e-3, key=0
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key)
    model = model_init_fn(keys[0])
    flat_params, apply_fn = model["flat_params"], model["apply_fn"]
    tx = optax.sgd(learning_rate=learning_rate)
    opt_state = tx.init(flat_params)
    train_state = TrainState(
        step=0, apply_fn=apply_fn, params=flat_params, tx=tx,
        opt_state=opt_state
    )
    def sgd_loss(params, x, y, apply_fn):
        prediction = vmap(apply_fn, (None, 0))(params, x).ravel()
        loss = jnp.sqrt(jnp.mean((prediction - y.ravel()) ** 2))
        
        return loss
    state, losses = sgd.train_full(
        key=keys[1], num_epochs=num_epochs, batch_size=batch_size,
        state=train_state, X=X, y=y, loss=sgd_loss,
    )
    
    return state, losses


def tune_and_store_hyperparameters(
    agent_type, hparam_path, model_init_fn, X, y, 
    n_explore=10, n_exploit=10,
):  
    print(f"Tuning {agent_type} hyperparameters...")
    pbounds = PBOUNDS[agent_type]
    if agent_type == "sgd":
        def bbf_offline_sgd(log_learning_rate):
            lr = jnp.exp(log_learning_rate).item()
            _, losses = train_offline_sgd(model_init_fn, X, y, learning_rate=lr)
            losses = losses["train"]
            mean_loss = jnp.mean(losses[-100:]).item()
            neg_loss = -mean_loss
            if jnp.isnan(neg_loss) or jnp.isinf(neg_loss):
                neg_loss = -1e8
            
            return neg_loss
        optimizer = BayesianOptimization(
            f=bbf_offline_sgd, pbounds=pbounds, verbose=2, random_state=1
        )
        optimizer.maximize(init_points=n_explore, n_iter=n_exploit)
        lr = jnp.exp(optimizer.max["params"]["log_learning_rate"]).item()
        best_hparams = {"learning_rate": lr}
    else:
        dataset = (X, y)
        rmse_callback = partial(
            callbacks.cb_eval,
            evaluate_fn = lambda w, apply_fn, x, y: \
                {
                    "rmse": -jnp.sqrt(jnp.mean((y - vmap(apply_fn, (None, 0))(w, x))**2)),
                }
        )
        optimizer = hparam_tune.create_optimizer(
            model_init_fn, pbounds, dataset, dataset,
            callback=rmse_callback, method="fcekf",
            callback_at_end=False,
        )
        optimizer.maximize(init_points=n_explore, n_iter=n_exploit)
        best_hparams = hparam_tune.get_best_params(optimizer, "fcekf")
    # Store as json
    with open(hparam_path, "w") as f:
        json.dump(best_hparams, f)

    return best_hparams


def main(cl_args):
    # Training dataset
    train_fn = _f1 if cl_args.train_dataset_type == 1 else _f2
    eval_fn =  _f1 if cl_args.eval_dataset_type == 1 else _f2
    X_train, y_train = generate_dataset(cl_args.n_train, 0.1, train_fn, 0, 
                                        in_between=True)
    
    # Eval dataset
    X_eval, y_eval = generate_dataset(cl_args.n_train, 0.1, eval_fn, 1,
                                      in_between=True)
    
    # Test dataset
    X_test = jnp.linspace(-5, 5, 200)
    
    # Initialize model
    model_init_fn = partial(models.initialize_regression_mlp,
                            input_dim=(1,), hidden_dims=cl_args.mlp_features,
                            emission_cov=0.1)
    model = model_init_fn(0)
    
    # Tune hyperparameters
    agent_type = "sgd" if "neural" in cl_args.agent else cl_args.agent
    config_path = Path("configs")
    Path(config_path).mkdir(parents=True, exist_ok=True)
    hparam_path = f"dataset_{cl_args.train_dataset_type}_" + \
        f"mlp_{cl_args.mlp_features}_n_train_{cl_args.n_train}_{agent_type}"
    hparam_path = Path(config_path, f"{hparam_path}.json")
    
    if cl_args.hyperparameters != "eval_only":
        agent_hparams = \
            tune_and_store_hyperparameters(
                agent_type, hparam_path, model_init_fn, X_train, y_train,
            )
    else:
        try:
            agent_hparams = json.load(open(hparam_path, "r"))
        except FileNotFoundError:
            raise ValueError(f"Hyperparameters for {agent_type} not found!")
        
    # Output
    output_path = Path("outputs")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if agent_type == "fcekf":
        estimator = hparam_tune.build_estimator(
            model_init_fn, agent_hparams, "fcekf", False
        )
        model = model_init_fn()
        # Train
        bel, _ = estimator["agent"].scan(
            model["flat_params"], estimator["init_cov"],
            X_train, y_train,
        )
        
        # Evaluate
        y_preds = vmap(model["apply_fn"], (None, 0))(bel.mean, X_test).ravel()
        y_cov = vmap(
            estimator["agent"].predict_obs_cov, (None, 0)
        )(bel, X_test).ravel()
        y_stds = jnp.sqrt(y_cov)
    elif agent_type == "sgd":
        ood = "_ood_" if cl_args.eval_ood else ""
        hparam_path_name = f"pretrain_{cl_args.train_dataset_type}_" + \
            f"eval_{cl_args.eval_dataset_type}_" + \
            f"mlp_{cl_args.mlp_features}_n_train_{cl_args.n_train}{ood}"
            
        # Train
        learning_rate = agent_hparams["learning_rate"]
        state, _ = train_offline_sgd(model_init_fn, X_train, y_train,
                                     learning_rate=learning_rate)
        
        # Neural-linear
        if "neural-linear" in cl_args.agent:
            y_preds = vmap(model["apply_fn"], (None, 0))(state.params, X_test).ravel()
            if cl_args.agent == "full-neural-linear":
                hparam_path_name += "_full_neural_linear"
                hparam_path = Path(config_path, hparam_path_name)
                n_features = len(cl_args.mlp_features)
                abridged_models = [
                    generate_sub_networks(model_init_fn, state.params, 0, i+1)
                    for i in range(n_features)
                ]
                phi_train = jnp.concatenate([
                    vmap(abridged_model["apply_fn"])(X_train)
                    for abridged_model in abridged_models
                ], axis=1)
                phi_test = jnp.concatenate([
                    vmap(abridged_model["apply_fn"])(X_test)
                    for abridged_model in abridged_models
                ], axis=1)
            else:
                hparam_path_name += "_neural_linear"
                hparam_path = Path(config_path, hparam_path_name)
                abridged_model = generate_sub_networks(
                    model_init_fn, state.params, 0, 2
                )
                phi_train = vmap(abridged_model["apply_fn"])(X_train)
                phi_test = vmap(abridged_model["apply_fn"])(X_test)
                
            y_cov = phi_test @ jnp.linalg.pinv(phi_train.T @ phi_train/0.3 +
                                            jnp.eye(phi_train.shape[-1])) \
                                                @ phi_test.T
            y_cov += 1e-2 * jnp.eye(len(y_cov))
            y_stds = jnp.sqrt(jnp.diag(y_cov))
        
        # Neural-stitching
        if cl_args.agent == "neural-stitching":
            hparam_path_name += f"_neural_stitching_{cl_args.stitch_layer}.json"
            hparam_path = Path(config_path, hparam_path_name)
            if cl_args.stitching_hyperparameters != "eval_only":
                model1 = generate_sub_networks(
                    model_init_fn, state.params, 0, cl_args.stitch_layer
                )
                model2 = generate_sub_networks(
                    model_init_fn, state.params, cl_args.stitch_layer, -1
                )
                stitch_model_init_fn = lambda key=0: \
                    construct_linear_stitching(model1, model2, key=key)

                stitch_agent_hparams = \
                    tune_and_store_hyperparameters(
                        "fcekf", hparam_path, stitch_model_init_fn, 
                        X_eval, y_eval,
                    )
            else:
                try:
                    stitch_agent_hparams = json.load(open(hparam_path, "r"))
                except FileNotFoundError:
                    raise ValueError(f"Hyperparameters for {cl_args.agent} not found!")
            
            # Evaluate
            stitch_estimator = hparam_tune.build_estimator(
                stitch_model_init_fn, stitch_agent_hparams, "fcekf", False
            )
            stitch_model = stitch_model_init_fn()
            stitch_bel, _ = stitch_estimator["agent"].scan(
                stitch_model["flat_params"], stitch_estimator["init_cov"],
                X_eval, y_eval,
            )
            y_preds = vmap(stitch_model["apply_fn"], (None, 0))(
                stitch_bel.mean, X_test
            ).ravel()
            y_cov = vmap(
                stitch_estimator["agent"].predict_obs_cov, (None, 0)
            )(stitch_bel, X_test).ravel()
            y_stds = jnp.sqrt(y_cov)
    else:
        raise ValueError(f"Agent {cl_args.agent} not supported!")
    
    # Save result
    result = {
        "y_preds": y_preds,
        "y_stds": y_stds,
    }
    result_path = Path(output_path, f"{hparam_path.stem}.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Number of training examples
    parser.add_argument("--n_train", type=int, default=200)
    
    # Training dataset
    parser.add_argument("--train_dataset_type", type=int, default=1,
                        choices=[1, 2])
    
    # Evaluation dataset
    parser.add_argument("--eval_dataset_type", type=int, default=1,
                        choices=[1, 2])
    
    # MLP hidden dimensions
    parser.add_argument("--mlp_features", type=_check_positive_int, nargs="+",
                        default=[50, 50, 50])
    
    # List of agents to use
    parser.add_argument("--agent", type=str, default="neural-linear",
                        choices=AGENT_TYPES)
    
    # Layer to stitch
    parser.add_argument("--stitch_layer", type=_check_positive_int, default=1)
    
    # Tune the hyperparameters of the agents
    parser.add_argument("--hyperparameters", type=str, default="tune_and_eval",
                        choices=["tune_and_eval", "tune_only", "eval_only"])
    
    # Tune the hyperparameters of the stitching agents
    parser.add_argument("--stitching_hyperparameters", type=str, 
                        default="tune_and_eval",
                        choices=["tune_and_eval", "tune_only", "eval_only"])
    
    # Whether to eval on OOD
    parser.add_argument("--eval_ood", action="store_true")
    
    args = parser.parse_args()
    main(args)

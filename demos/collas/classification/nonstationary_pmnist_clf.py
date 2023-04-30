import os
from pathlib import Path
from functools import partial

from flax.training.train_state import TrainState
import jax
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import optax

from demos.showdown.classification import classification_train as benchmark
from demos.showdown.classification import hparam_tune_clf as hpt
from rebayes.datasets import classification_data as data_utils
from rebayes.sgd_filter import sgd


# def generate_warmup_data(data_dict, ntrain_per_task, nval_per_task, val_after=50):
#     (Xtr, Ytr), (Xval, Yval), _ = data_dict.values()
#     Xtr_warmup, Ytr_warmup = Xtr[:val_after*ntrain_per_task], Ytr[:val_after*ntrain_per_task]
#     warmup_train = (Xtr_warmup, Ytr_warmup)

#     Xval_warmup, Yval_warmup = Xval[:val_after*nval_per_task], Yval[:val_after*nval_per_task]
#     warmup_val = (Xval_warmup, Yval_warmup)
    
#     return warmup_train, warmup_val


def train_agent(
    ntrain_per_task,
    ntest_per_task,
    model_dict, 
    dataset, 
    load_dataset_fn, 
    agent_type='fdekf', 
    **kwargs
):
    print(f'Training {agent_type} agent...')
    model, emission_mean_function, emission_cov_function = \
        model_dict['model'], model_dict['emission_mean_function'], model_dict['emission_cov_function']
    
    if 'sgd' in agent_type:
        pbounds = {
            'log_learning_rate': (-10, 0.0),
        }
        init_points, n_iter = 5, 5
    else:
        pbounds={
            'log_init_cov': (-10, 0),
            'log_dynamics_weights': (-40, 0),
            'log_dynamics_cov': (-40, 0),
            'log_alpha': (-90, -90),
        }
        if 'lofi' in agent_type:
            agent_type = 'lofi'
        init_points, n_iter = 20, 25
        
    ll_callback = partial(
        benchmark.osa_eval_callback, 
        evaluate_fn=lambda y_pred, y: -optax.softmax_cross_entropy(y_pred, y).mean(),
    )
    optimizer, *_ = hpt.create_optimizer(
        model, pbounds, 0, dataset['train'], dataset['val'], emission_mean_function,
        emission_cov_function, callback=ll_callback, method=agent_type, verbose=2,
        callback_at_end=False, **kwargs
    )
    
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )
    best_hparams = hpt.get_best_params(optimizer, method=agent_type)
    print(f"Best target: {optimizer.max['target']}")
    
    estimator = hpt.build_estimator(
        model_dict['flat_params'],
        model_dict['apply_fn'],
        best_hparams,
        emission_mean_function,
        emission_cov_function,
        method=agent_type,
        **kwargs,
    )
    
    # per_batch_miscl_callback = partial(benchmark.per_batch_callback, evaluate_fn=benchmark.mnist_evaluate_miscl)
    result = jax.block_until_ready(
        benchmark.nonstationary_mnist_eval_agent(
            load_dataset_fn,
            ntrain_per_task,
            ntest_per_task,
            model_dict['apply_fn'],
            estimator,
            n_iter=20,
        )
    )
    nll_result, miscl_result = (
        {
            "current_mean": result[key]["current"].mean(axis=0),
            "current_std": result[key]["current"].std(axis=0),
            "task1_mean": result[key]["task1"].mean(axis=0),
            "task1_std": result[key]["task1"].std(axis=0),
            "overall_mean": result[key]["overall"].mean(axis=0),
            "overall_std": result[key]["overall"].std(axis=0),
        }
        for key in ('nll', 'miscl')
    )
    
    return nll_result, miscl_result


def evaluate_offline_sgd(model_dict, dataset, num_epochs=10_000, batch_size=32, key=0):
    print('Training Baseline Offline SGD agent...')
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    Xtr, Ytr = dataset['train']
    Xte, Yte = dataset['test']
    
    sgd_state = TrainState.create(
        apply_fn=model_dict['apply_fn'],
        params=model_dict['flat_params'],
        tx=optax.sgd(learning_rate=1e-4)
    )

    @partial(jit, static_argnums=(3,))
    def sgd_loss_fn(params, X_batch, y_batch, apply_fn):
        logits = vmap(apply_fn, (None, 0))(params, X_batch).ravel()
        nll = jnp.mean(optax.softmax_cross_entropy(logits, y_batch.ravel()))

        return nll
    
    sgd_state, losses = sgd.train_full(
        key, num_epochs, batch_size, sgd_state, 
        Xtr, Ytr, sgd_loss_fn, Xte, jax.nn.one_hot(Yte, 10)
    )
    nll_loss_fn = lambda logits, label: optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
    miscl_loss_fn = lambda logits, label: jnp.mean(logits.argmax(axis=-1) != label)
    nll_evaluate_fn = partial(benchmark.evaluate_function, loss_fn=nll_loss_fn)
    miscl_evaluate_fn = partial(benchmark.evaluate_function, loss_fn=miscl_loss_fn)
    
    nll_result = nll_evaluate_fn(sgd_state.params, model_dict['apply_fn'], Xte, Yte)
    miscl_result = miscl_evaluate_fn(sgd_state.params, model_dict['apply_fn'], Xte, Yte)
    print(f"Offline SGD NLL: {nll_result}, Misclassification: {miscl_result}")
    
    return nll_result, miscl_result
    

if __name__ == "__main__":
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        output_path = Path(Path.cwd(), "output", "nonstationary", "permuted_mnist")
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")
    
    data_kwargs = {
        'n_tasks': 10,
        'ntrain_per_task': 300,
        'nval_per_task': 1,
        'ntest_per_task': 1_000,
    }
    dataset = data_utils.load_permuted_mnist_dataset(**data_kwargs, fashion=True) # load data
    dataset_load_fn = partial(data_utils.load_permuted_mnist_dataset, **data_kwargs, fashion=True)
    
    features = [500, 500, 10]
    model_dict = benchmark.init_model(type='mlp', features=features)
    
    # Evaluate offline SGD baseline
    offline_sgd_nll, offline_sgd_miscl = evaluate_offline_sgd(model_dict, dataset)
    benchmark.store_results(offline_sgd_nll, "offline_sgd_nll", output_path)
    benchmark.store_results(offline_sgd_miscl, "offline_sgd_miscl", output_path)
    
    # Evaluate online methods
    lofi_ranks = (
        # 1,
        # 5,
        10,
    )
    lofi_methods = (
        # "spherical", 
        "diagonal",
    )
    lofi_agents = {
        f'lofi-{rank}-{method}': {
            'memory_size': rank,
            'inflation': "hybrid",
            'lofi_method': method,
        } for rank in lofi_ranks for method in lofi_methods
    }
    
    sgd_optimizer = (
        "sgd",
        "adam",
    )
    sgd_ranks = (
        1,
        # 5,
        10,
    )
    sgd_agents = {
        f'sgd-rb-{rank}-{optimizer}': {
            'loss_fn': optax.softmax_cross_entropy,
            'buffer_size': rank,
            'dim_output': 10,
            "optimizer": optimizer,
        } for rank in sgd_ranks for optimizer in sgd_optimizer
    }
    
    agents = {
        **sgd_agents,
        **lofi_agents,
        'fdekf': None,
        'vdekf': None,
    }
    
    nll_results, miscl_results = {}, {}
    for agent, kwargs in agents.items():
        if kwargs is None:
            nll, miscl = train_agent(
                data_kwargs["ntrain_per_task"],
                data_kwargs["ntest_per_task"],
                model_dict, 
                dataset, 
                dataset_load_fn, 
                agent_type=agent
            )
        else:
            nll, miscl = train_agent(
                data_kwargs["ntrain_per_task"],
                data_kwargs["ntest_per_task"],
                model_dict, 
                dataset, 
                dataset_load_fn, 
                agent_type=agent,
                **kwargs
            )
        benchmark.store_results(nll, f'{agent}_nll', output_path)
        benchmark.store_results(miscl, f'{agent}_miscl', output_path)
        nll_results[agent] = nll
        miscl_results[agent] = miscl
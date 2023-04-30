from functools import partial
import os
from pathlib import Path

import jax
import optax

from demos.showdown.classification import (
    classification_train as benchmark,
    hparam_tune_clf as hpt
)
from rebayes.datasets import classification_data as data_utils


def train_agent(model_dict, dataset, agent_type='fdekf', **kwargs):
    print(f'Training {agent_type} agent...')
    X_train, y_train = dataset['train']
    train = (X_train[:500], y_train[:500])
    model, emission_mean_function, emission_cov_function = \
        model_dict['model'], model_dict['emission_mean_function'], model_dict['emission_cov_function']
    
    if 'sgd' in agent_type:
        pbounds = {
            'log_learning_rate': (-10, 0.0),
        }
        init_points, n_iter = 1, 1
    else:
        pbounds={
            'log_init_cov': (-10, 0.0),
            'log_dynamics_weights': (-90, -90),
            'log_dynamics_cov': (-90, -90),
            'log_alpha': (-40, 0),
        }
        if 'lofi' in agent_type:
            agent_type = 'lofi'
        init_points, n_iter = 20, 25
    
    ll_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_ll)
    optimizer, *_ = hpt.create_optimizer(
        model, pbounds, 0, train, dataset['val'], emission_mean_function,
        emission_cov_function, callback=ll_callback, method=agent_type, verbose=1,
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
    
    eval_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_nll_and_miscl)
    
    result, runtime = jax.block_until_ready(
        benchmark.mnist_eval_agent(
            dataset['train'], dataset['test'], model_dict['apply_fn'], 
            callback=eval_callback, agent=estimator, n_iter=100
        )
    )
    
    nll_result, miscl_result = (
        {
            "mean": result[key].mean(axis=0), 
            "std": result[key].std(axis=0), 
            'runtime': runtime,
        }
        for key in ('nll', 'miscl')
    )
    
    return nll_result, miscl_result


if __name__ == "__main__":
    fashion = True
    
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        dataset_name = "mnist" if not fashion else "f-mnist"
        output_path = Path(Path.cwd(), "output", "stationary")
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")
    
    dataset = data_utils.load_mnist_dataset(fashion=fashion) # load data
    # model_dict = benchmark.init_model(type='mlp', features=(500, 500, 10)) # initialize model
    model_dict = benchmark.init_model(type='cnn') # initialize model
    
    lofi_ranks = (
        1,
        5,
        10,
        20,
        50,
    )
    lofi_methods = ("spherical", "diagonal")
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
        **lofi_agents,
        'fdekf': None,
        'vdekf': None,
        **sgd_agents,
    }
    
    nll_results, miscl_results = {}, {}
    for agent, kwargs in agents.items():
        if kwargs is None:
            nll, miscl = train_agent(model_dict, dataset, agent_type=agent)
        else:
            nll, miscl = train_agent(model_dict, dataset, agent_type=agent, **kwargs)
        benchmark.store_results(nll, f'{agent}_nll', output_path)
        benchmark.store_results(miscl, f'{agent}_miscl', output_path)
        nll_results[agent] = nll
        miscl_results[agent] = miscl
    
    nll_title = "Test-set average negative log likelihood"
    benchmark.plot_results(nll_results, "mnist_nll", output_path, ylim=(0.5, 2.5), title=nll_title)
    
    miscl_title = "Test-set average misclassification rate"
    benchmark.plot_results(miscl_results, "mnist_miscl", output_path, ylim=(0.0, 1.0), title=miscl_title)
    
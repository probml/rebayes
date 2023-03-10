import os
from pathlib import Path
from functools import partial

import jax
import optax

from rebayes.low_rank_filter.lofi import LoFiParams
from demos.showdown.classification import classification_train as benchmark
from demos.showdown.classification import hparam_tune_clf as hpt


def train_agent(model_dict, dataset, agent_type='fdekf', **kwargs):
    print(f'Training {agent_type} agent...')
    X_train, y_train = dataset['train']
    train = (X_train[:500], y_train[:500])
    model, emission_mean_function, emission_cov_function = \
        model_dict['model'], model_dict['emission_mean_function'], model_dict['emission_cov_function']
    
    if 'sgd' in agent_type:
        pbounds = {
            'learning_rate': (1e-6, 1e-2),
        }
        init_points, n_iter = 10, 15
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
        model, pbounds, 314, train, dataset['val'], emission_mean_function,
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
    
    miscl_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_miscl)
    miscl_mean, miscl_std = benchmark.mnist_eval_agent(
        dataset['train'], dataset['test'], model_dict['apply_fn'], callback=miscl_callback, agent=estimator,
        n_iter=20
    )

    miscl_result = jax.block_until_ready({
        'mean': miscl_mean,
        'std': miscl_std,
    })
    print('\n')
    
    return miscl_result


if __name__ == "__main__":
    fashion = True
    
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        dataset_name = "mnist" if not fashion else "f-mnist"
        output_path = Path(Path.cwd(), "output", "final", "stationary")
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")
    
    dataset = benchmark.load_mnist_dataset(fashion=fashion) # load data
    model_dict = benchmark.init_model(type='cnn') # initialize model
    
    lofi_ranks = (1, 2, 5, 10, 20, 50)
    lofi_agents = {
        f'lofi-{rank}': {
            'lofi_params': LoFiParams(memory_size=rank, diagonal_covariance=True),
            'inflation': 'hybrid',
        } for rank in lofi_ranks
    }
    
    sgd_ranks = (1, 10, 20)
    sgd_agents = {
        f'sgd-rb-{rank}': {
            'loss_fn': optax.softmax_cross_entropy,
            'buffer_size': rank,
            'dim_output': 10,
        } for rank in sgd_ranks
    }
    
    agents = {
        **sgd_agents,
        'fdekf': None,
        'vdekf': None,
        **lofi_agents,
    }
    
    miscl_results = {}
    for agent, kwargs in agents.items():
        if kwargs is None:
            miscl = train_agent(model_dict, dataset, agent_type=agent)
        else:
            miscl = train_agent(model_dict, dataset, agent_type=agent, **kwargs)
        benchmark.store_results(miscl, f'{agent}_miscl', output_path)
        miscl_results[agent] = miscl
        
    # Store results and plot
    benchmark.store_results(miscl_results, 'mnist_miscl', output_path)
    
    miscl_title = "Test-set average misclassification rate"
    benchmark.plot_results(miscl_results, "mnist_miscl", output_path, ylim=(0.0, 0.8), title=miscl_title)
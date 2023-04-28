import os
from pathlib import Path
from functools import partial

import jax
import optax

from rebayes.low_rank_filter.lofi import LoFiParams
from demos.showdown.classification import classification_train as benchmark
from demos.showdown.classification import hparam_tune_clf as hpt


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
            'learning_rate': (1e-6, 1e-2),
        }
        init_points, n_iter = 10, 15
    else:
        pbounds={
            'log_init_cov': (-10, 0),
            'log_dynamics_weights': (-40, 0),
            'log_dynamics_cov': (-40, 0),
            'log_alpha': (-40, 0),
        }
        if 'lofi' in agent_type:
            agent_type = 'lofi'
        init_points, n_iter = 10, 15
        
    ll_callback = partial(
        benchmark.osa_eval_callback, 
        evaluate_fn=lambda y_pred, y: -optax.softmax_cross_entropy(y_pred, y).mean(),
    )
    optimizer, *_ = hpt.create_optimizer(
        model, pbounds, 0, dataset['train'], dataset['val'], emission_mean_function,
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
    
    # per_batch_miscl_callback = partial(benchmark.per_batch_callback, evaluate_fn=benchmark.mnist_evaluate_miscl)
    miscl_result = jax.block_until_ready(
        benchmark.nonstationary_mnist_eval_agent(
            load_dataset_fn,
            ntrain_per_task,
            ntest_per_task,
            model_dict['apply_fn'],
            estimator,
            n_iter=10,
        )
    )
    print('\n')
    
    return miscl_result


if __name__ == "__main__":
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        output_path = Path(Path.cwd(), "output", "final", "nonstationary", "permuted_mnist")
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")
    
    data_kwargs = {
        'n_tasks': 10,
        'ntrain_per_task': 300,
        'nval_per_task': 1_000,
        'ntest_per_task': 1_000,
    }
    dataset = benchmark.load_permuted_mnist_dataset(**data_kwargs, fashion=True) # load data
    dataset_load_fn = partial(benchmark.load_permuted_mnist_dataset, **data_kwargs, fashion=True)
    
    # warmup_train, warmup_val = generate_warmup_data(
    #     dataset, data_kwargs["ntrain_per_task"], data_kwargs["nval_per_task"], val_after=50
    # )
    # dataset['warmup_train'] = warmup_train
    # dataset['warmup_val'] = warmup_val
    
    features = [100, 100, 10]
    model_dict = benchmark.init_model(type='mlp', features=features)
    
    lofi_ranks = (20,)
    lofi_agents = {
        f'lofi-{rank}': {
            'memory_size': rank,
            'inflation': "hybrid",
        } for rank in lofi_ranks
    }
    
    sgd_ranks = (1, 20)
    sgd_agents = {
        f'sgd-rb-{rank}': {
            'loss_fn': optax.softmax_cross_entropy,
            'buffer_size': rank,
            'dim_output': 10,
        } for rank in sgd_ranks
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
            nll = train_agent(
                data_kwargs["ntrain_per_task"],
                data_kwargs["ntest_per_task"],
                model_dict, 
                dataset, 
                dataset_load_fn, 
                agent_type=agent
            )
        else:
            nll = train_agent(
                data_kwargs["ntrain_per_task"],
                data_kwargs["ntest_per_task"],
                model_dict, 
                dataset, 
                dataset_load_fn, 
                agent_type=agent,
                **kwargs
            )
        benchmark.store_results(nll, f'{agent}_mnist_nll', output_path)
        nll_results[agent] = nll
        
    # Store results and plot
    benchmark.store_results(nll_results, 'mnist_nll', output_path)
    
    # nll_title = "Test-set average negative log likelihood"
    # benchmark.plot_results(nll_results, "mnist_nll", output_path, ylim=(0.5, 2.5), title=nll_title)
import os
from pathlib import Path
from functools import partial

import jax
import optax

from rebayes.low_rank_filter.lofi import LoFiParams
from demos.showdown.classification import classification_train as benchmark
from demos.showdown.classification import hparam_tune_clf as hpt


def generate_warmup_data(data_dict, ntrain_per_task, nval_per_task, val_after=5):
    (Xtr, Ytr), (Xval, Yval), _ = data_dict.values()
    Xtr_warmup, Ytr_warmup = Xtr[:val_after*ntrain_per_task], Ytr[:val_after*ntrain_per_task]
    warmup_train = (Xtr_warmup, Ytr_warmup)

    Xval_warmup = Xval[(val_after-1)*nval_per_task:val_after*nval_per_task]
    Yval_warmup = Yval[(val_after-1)*nval_per_task:val_after*nval_per_task]
    warmup_val = (Xval_warmup, Yval_warmup)
    
    return warmup_train, warmup_val


def train_agent(model_dict, dataset, agent_type='fdekf', **kwargs):
    print(f'Training {agent_type} agent...')
    model, emission_mean_function, emission_cov_function = \
        model_dict['model'], model_dict['emission_mean_function'], model_dict['emission_cov_function']
    
    if 'sgd' in agent_type:
        pbounds = {
            'log_lr': (-8.0, 0.0),
        }
    else:
        pbounds={
            'log_init_cov': (-10, 5.0),
            'log_dynamics_weights': (-90, -90),
            'log_dynamics_cov': (-90, -90),
            'log_alpha': (-40, 0),
        }
        if 'lofi' in agent_type:
            agent_type = 'lofi'
    
    ll_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_ll)
    optimizer, *_ = hpt.create_optimizer(
        model, pbounds, 314, dataset['warmup_train'], dataset['warmup_val'], emission_mean_function,
        emission_cov_function, callback=ll_callback, method=agent_type, verbose=1, **kwargs
    )
    
    optimizer.maximize(
        init_points=20,
        n_iter=25,
    )
    best_hparams = hpt.get_best_params(optimizer, method=agent_type)
    print(f"Best target: {optimizer.max['target']}")
    
    estimator, bel = hpt.build_estimator(
        model_dict['flat_params'],
        model_dict['apply_fn'],
        best_hparams,
        emission_mean_function,
        emission_cov_function,
        method=agent_type,
        **kwargs,
    )
    
    nll_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_nll)
    nll_mean, nll_std = benchmark.mnist_eval_agent(
        dataset['train'], dataset['test'], model_dict['apply_fn'], callback=nll_callback,
        agent=estimator, bel_init=bel,
    )
    
    miscl_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_miscl)
    miscl_mean, miscl_std = benchmark.mnist_eval_agent(
        dataset['train'], dataset['test'], model_dict['apply_fn'], callback=miscl_callback,
        agent=estimator, bel_init=bel,
    )
    
    nll_result = jax.block_until_ready({
        'mean': nll_mean,
        'std': nll_std,
    })
    miscl_result = jax.block_until_ready({
        'mean': miscl_mean,
        'std': miscl_std,
    })
    print('\n')
    
    return nll_result, miscl_result


if __name__ == "__main__":
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        output_path = Path(Path.cwd(), "output", "permuted-mnist")
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")
    
    data_kwargs = {
        'n_tasks': 100,
        'ntrain_per_task': 600,
        'nval_per_task': 5_000,
        'ntest_per_task': 5_000,
    }
    dataset = benchmark.load_permuted_mnist_dataset(**data_kwargs, fashion=True) # load data
    warmup_train, warmup_val = generate_warmup_data(
        dataset, data_kwargs["ntrain_per_task"], data_kwargs["nval_per_task"], val_after=5
    )
    dataset['warmup_train'] = warmup_train
    dataset['warmup_val'] = warmup_val
    
    features = [400, 400, 10]
    model_dict = benchmark.init_model(type='mlp', features=features)
    
    lofi_ranks = (1, 5, 10, 20, 50)
    lofi_agents = {
        f'lofi-{rank}': {
            LoFiParams(memory_size=rank, diagonal_covariance=False)
        } for rank in lofi_ranks
    }
    
    sgd_ranks = (1, 5, 10, 20, 50)
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
    
    nll_results, miscl_results = {}, {}
    for agent, kwargs in agents.items():
        if kwargs is None:
            nll, miscl = train_agent(model_dict, dataset, agent_type=agent)
        else:
            nll, miscl = train_agent(model_dict, dataset, agent_type=agent, **kwargs)
        benchmark.store_results(nll, f'{agent}_mnist_nll', output_path)
        benchmark.store_results(miscl, f'{agent}_mnist_miscl', output_path)
        nll_results[agent] = nll
        miscl_results[agent] = miscl
        
    # Store results and plot
    benchmark.store_results(nll_results, 'mnist_nll', output_path)
    benchmark.store_results(miscl_results, 'mnist_miscl', output_path)
    
    nll_title = "Test-set average NLL"
    benchmark.plot_results(nll_results, "mnist_nll", output_path, ylim=(0.5, 2.5), title=nll_title)
    
    miscl_title = "Test-set average misclassification rate"
    benchmark.plot_results(miscl_results, "mnist_miscl", output_path, ylim=(0.0, 0.8), title=miscl_title)
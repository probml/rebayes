from functools import partial
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from demos.collas.regression.rotating_mnist_unsorted import increase_damp_angle_bounded
from demos.showdown.classification import (
    classification_train as benchmark,
    hparam_tune_clf as hpt
)
from rebayes.datasets import classification_data as data_utils
from rebayes.datasets.rotating_permuted_mnist_data import (
    rotate_mnist_dataset,
    generate_rotating_mnist_dataset
)


def train_agent(model_dict, dataset, agent_type='fdekf', gr_val=True, **kwargs):
    print(f'Training {agent_type} agent...')
    X_train, y_train = dataset['train']
    eval_train = (X_train, y_train)
    X_val, y_val = dataset['val']
    X_test, y_test = dataset['test']
    n_steps = min(1_000, len(X_val))
    min_angle, max_angle = 0.0, 360.0
    
    # Gradually damped-rotate training set
    gradually_rotating_angles = increase_damp_angle_bounded(n_steps, min_angle, max_angle)
    X_train = rotate_mnist_dataset(X_train[:n_steps], gradually_rotating_angles)
    y_train = y_train[:n_steps]
    train = (X_train, y_train)

    if gr_val:
        # Gradually rotate validation set
        X_val = rotate_mnist_dataset(X_val[:n_steps], gradually_rotating_angles)
        ll_callback = partial(benchmark.window_callback_loss, loss_fn=optax.softmax_cross_entropy_with_integer_labels,)
        eval_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_nll_and_miscl)
        
        # Gradually rotate test set
        X_test = rotate_mnist_dataset(X_test[:n_steps], gradually_rotating_angles)
        y_test = y_test[:n_steps]
    else:
        # Randomly rotate validation set
        X_val, _ = generate_rotating_mnist_dataset(X_val[:n_steps], min_angle, max_angle)
        ll_callback = partial(benchmark.eval_callback, evaluate_fn=benchmark.mnist_evaluate_ll)
        eval_callback = partial(benchmark.window_callback_eval, evaluate_fn=benchmark.mnist_evaluate_nll_and_miscl)
        
        # Randomly rotate test set
        X_test, _ = generate_rotating_mnist_dataset(X_test, min_angle, max_angle)
    y_val = y_val[:n_steps]
    val = (X_val, y_val)
    test = (X_test, y_test)

    model, emission_mean_function, emission_cov_function = \
        model_dict['model'], model_dict['emission_mean_function'], model_dict['emission_cov_function']
    
    if 'sgd' in agent_type:
        pbounds = {
            'log_learning_rate': (-10, 0.0),
        }
        init_points, n_iter = 5, 5
    else:
        pbounds={
            'log_init_cov': (-10, 0.0),
            'log_dynamics_weights': (-40, 0),
            'log_dynamics_cov': (-40, 0),
            'log_alpha': (-90, -90),
        }
        if 'lofi' in agent_type:
            agent_type = 'lofi'
        init_points, n_iter = 20, 25
    
    optimizer, *_ = hpt.create_optimizer(
        model, pbounds, 0, train, val, emission_mean_function,
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
    
    result, runtime = jax.block_until_ready(
        benchmark.rotating_mnist_eval_agent(
            eval_train, test, model_dict['apply_fn'], 
            callback=eval_callback, agent=estimator, n_iter=100,
            n_steps=n_steps, min_angle=min_angle, max_angle=max_angle,
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
    for fashion in (
            True, 
            # False,
        ):
        for gr_val in (
                True, 
                # False,
            ):
            output_path = os.environ.get("REBAYES_OUTPUT")
            if output_path is None:
                dataset_name = "mnist" if not fashion else "fmnist"
                rot_type = "gradually_rotating" if gr_val else "randomly_rotated"
                output_path = Path(Path.cwd(), "output", "nonstationary", f"damped_rotating_{dataset_name}", f"{rot_type}_val")
                output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output path: {output_path}")
            
            dataset = data_utils.load_mnist_dataset(fashion=fashion) # load data
            model_dict = benchmark.init_model(type='mlp', features=(500, 500, 10)) # initialize model
            
            lofi_ranks = (
                # 1,
                # 5,
                10,
                # 20,
                # 50,
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
                **lofi_agents,
                'fdekf': None,
                'vdekf': None,
                **sgd_agents,
            }
            
            nll_results, miscl_results = {}, {}
            for agent, kwargs in agents.items():
                if kwargs is None:
                    nll, miscl = train_agent(model_dict, dataset, agent_type=agent, gr_val=gr_val)
                else:
                    nll, miscl = train_agent(model_dict, dataset, agent_type=agent, gr_val=gr_val, **kwargs)
                benchmark.store_results(nll, f'{agent}_nll', output_path)
                benchmark.store_results(miscl, f'{agent}_miscl', output_path)
                nll_results[agent] = nll
                miscl_results[agent] = miscl
            
            nll_title = "Test-set average negative log likelihood"
            benchmark.plot_results(nll_results, "mnist_nll", output_path, ylim=(0.5, 2.5), title=nll_title)
            
            miscl_title = "Test-set average misclassification rate"
            benchmark.plot_results(miscl_results, "mnist_miscl", output_path, ylim=(0.0, 1.0), title=miscl_title)
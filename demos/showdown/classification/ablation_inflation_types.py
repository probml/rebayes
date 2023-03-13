import os
from pathlib import Path
from functools import partial

import jax
import optax

from rebayes.low_rank_filter.lofi import LoFiParams
from demos.showdown.classification import classification_train as benchmark
from demos.showdown.classification import hparam_tune_clf as hpt
from demos.showdown.classification.mnist import train_agent


INFLATION_TYPES = ("bayesian", "hybrid", "simple")


def train_lofi_ablation_type(infl_type="bayesian"):
    if infl_type not in INFLATION_TYPES:
        raise ValueError(f"Unknown inflation type: {infl_type}")
    
    lofi_ranks = (1, 5,)
    lofi_agents = {
        f'lofi-{rank}': {
            'lofi_params': LoFiParams(memory_size=rank, diagonal_covariance=True),
            'inflation': infl_type,
        } for rank in lofi_ranks
    }
    
    results = {}
    for agent, kwargs in lofi_agents.items():
        miscl = train_agent(model_dict, dataset, agent_type=agent, **kwargs)
        results[f'{agent}_miscl'] = miscl
    
    return results


if __name__ == "__main__":
    fashion = True
    
    output_path = os.environ.get("REBAYES_OUTPUT")
    if output_path is None:
        dataset_name = "mnist" if not fashion else "f-mnist"
        output_path = Path(Path.cwd(), "output", "stationary", "ablation")
        output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")
    
    dataset = benchmark.load_mnist_dataset(fashion=fashion) # load data
    model_dict = benchmark.init_model(type='mlp', features=(100, 100, 10)) # initialize model
    
    for infl_type in INFLATION_TYPES:
        curr_output_path = Path(output_path, infl_type)
        curr_output_path.mkdir(parents=True, exist_ok=True)
        
        results = train_lofi_ablation_type(infl_type)
        for key, val in results.items():
            benchmark.store_results(val, f'{key}_miscl', curr_output_path)
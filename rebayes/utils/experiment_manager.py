from typing import Sequence, Union
from abc import ABC
from abc import abstractmethod

import jax.numpy as jnp
from jax import vmap, jit
import optax
from bayes_opt import BayesianOptimization
from tqdm import trange
import matplotlib.pyplot as plt

from rebayes.extended_kalman_filter.ekf import RebayesEKF
from rebayes.low_rank_filter.lofi import RebayesLoFi
from rebayes.low_rank_filter.lofi_inference import LoFiParams
from rebayes.base import RebayesParams
from rebayes.utils.utils import get_mlp_flattened_params

from rebayes.utils.datasets import (
    load_rotated_mnist,
    load_classification_mnist,
    load_1d_synthetic_dataset,
    load_uci_wine_regression,
    load_uci_naval,
    load_uci_kin8nm,
    load_uci_power,
    load_uci_spam
)


class ExpManager(ABC):
    def __init__(
        self,
        dataset: str,
        model: str,
        method: str,
        eval_interval: float=0.0,
    ):
        available_datasets = {
            'synthetic_regression': load_1d_synthetic_dataset,
            'uci_wine_regression': load_uci_wine_regression,
            'uci_naval_regression': load_uci_naval,
            'uci_kin8nm_regression': load_uci_kin8nm,
            'uci_power_regression': load_uci_power,
            'mnist_regression': load_rotated_mnist,
            'mnist_classification': load_classification_mnist,
            # 'split_mnist_regression':
            # 'split_mnist_classification', # TODO
        }
        available_models = (
            'linear_mlp', # TODO
            'nonlinear_mlp', # TODO
            'cnn' # TODO
        )
        available_methods = (
            'orfit-10',
            'orfit-50',
            'orfit-100',
            'lofi-10',
            'lofi-50',
            'lofi-100',
            'aov_lofi-10',
            'aov_lofi-50',
            'aov_lofi-100',
        )
        assert isinstance(dataset, str) and dataset in available_datasets, \
            f"parameter 'dataset' must be one of the following: {available_datasets}."
        assert isinstance(model, str) and model in available_models, \
            f"parameter 'model' must be one of the following: {available_models}."
        assert eval_interval >= 0.0 and eval_interval <= 1.0, \
            "parameter 'eval_interval' must be a float between 0.0 and 1.0."
        if dataset == 'synthetic_regression':
            assert model != 'cnn', "cnn model not supported for 'synthetic_regression' dataset."
        assert method in available_methods, f'Invalid method: {method}'

        # Define method
        self.method = method
        self.method_title, self.memory = method.split('-')
        self.memory = int(self.memory)

        # Initialize NN model
        self.dataset = dataset
        self.model = model
        self.params, self.apply_fn = self.init_model()

        if dataset[-10:] == 'regression':
            self.loss = lambda y, y_pred: jnp.sqrt(jnp.mean((y - y_pred)**2)) # RMSE
            self.emission_cov_fn = None
        else:
            self.loss = lambda y, y_pred: optax.softmax_cross_entropy_with_integer_labels(logits=y, labels=y_pred) # classification error
            def emission_cov_function(w, x):
                ps = self.apply_fn(w, x)
                return jnp.diag(ps) - jnp.outer(ps, ps) + 1e-3 * jnp.eye(len(ps)) # Add diagonal to avoid singularity
            self.emission_cov_fn = emission_cov_function

        # Load dataset
        self.dataset_loader = available_datasets[dataset]
        (self.X_warmup, self.y_warmup), (self.X_train, self.y_train), \
            (self.X_test, self.y_test) = self.load_dataset()

        # Define eval interval
        self.eval_interval = max(1, int(len(self.X_train) * eval_interval))

    def init_model(self):
        input_dims = {
            'synthetic_regression': 1,
            'mnist_regression': 28,
            'mnist_classification': 28,
            'split_mnist_regression': 28,
            'split_mnist_classification': 28,
        }
        if self.model == 'linear_mlp':
            model_dims = [input_dims[self.dataset]**2, 1]
        elif self.model == 'nonlinear_mlp':
            model_dims = [input_dims[self.dataset]**2, 100, 100, 1]
        elif self.model == 'cnn':
            model_dims = [input_dims[self.dataset]]
        _, params, _, apply_fn = get_mlp_flattened_params(model_dims, model=self.model[-3:])
        
        return params, apply_fn

    def load_dataset(self, n_warmup=500):
        (X_train, y_train), (X_test, y_test) = self.dataset_loader()
        X_warmup, y_warmup = X_train[:n_warmup], y_train[:n_warmup]
        X_train, y_train = X_train[n_warmup:], y_train[n_warmup:]

        return (X_warmup, y_warmup), (X_train, y_train), (X_test, y_test)
    
    def filter_and_evaluate(self, estimator):
        """Run filter and evaluate."""
        # train and evaluate
        rmses = []
        bel = estimator.init_bel()
        for i in trange(self.X_train.shape[0]):
            # Update params
            bel = estimator.predict_state(bel)
            bel = estimator.update_state(bel, self.X_train[i], self.y_train[i])

            # Evaluate
            if i % self.eval_interval == 0:
                y_pred = vmap(jit(self.apply_fn), (None, 0))(bel.mean, self.X_test).squeeze()
                rmse = jnp.sqrt(((y_pred - self.y_test)**2).mean())
                rmses.append(rmse)
        
        return jnp.array(rmses)

    @staticmethod
    def _cross_validation_splits(n_folds, X, y):
        n = X.shape[0]
        fold_size = n // n_folds
        train_val_splits = [
            (
                jnp.concatenate((X[:i*fold_size], X[(i+1)*fold_size:])), 
                jnp.concatenate((y[:i*fold_size], y[(i+1)*fold_size:])),
                X[i*fold_size:(i+1)*fold_size],
                y[i*fold_size:(i+1)*fold_size],
            )
            for i in range(n_folds)
        ]
        return train_val_splits

    def plot_results(self, ax, result_dict, title, legend_loc=None):
        """Plot results."""
        for key, val in result_dict.items():
            ax.plot(val['rmse'], color=val['color'], ls=val['ls'], label=key)
            ax.fill_between(
                jnp.arange(val['rmse'].shape[0]),
                val['rmse'] - 10*val['rmse_std'],
                val['rmse'] + 10*val['rmse_std'],
                alpha=0.1, color=val['color']
            )

        if legend_loc is not None:
            ax.legend(loc=legend_loc)
        ax.grid()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 125)
        ax.set_xlabel('Training Steps (samples)')
        ax.set_ylabel('RMSE (deg)')
        ax.set_title(title)

        return ax
    
    @abstractmethod
    def choose_hparams(self):
        """Tune and return hyper-parameters using Bayesian Optimization."""
        raise NotImplementedError

    def run_experiment(self, ax):
        """Run experiment."""
        # Choose hyper-parameters
        estimator = self.choose_hparams()

        # Run experiment
        rmses = self.filter_and_evaluate(estimator)
        results = {
            self.method: {
                'rmse': rmses,
            }
        }
        ax = self.plot_results(
            ax, 
            results, 
            title=f'{self.dataset} Test Error',
            legend_loc='upper right'
        )
        
        return ax


class LoFiManager(ExpManager):
    def __init__(
        self,
        dataset: str,
        model: str,
        method: str,
        eval_interval: float=0.0,
    ):
        super().__init__(dataset, model, method, eval_interval)
    
    @staticmethod
    def _opt_fn(
        params, apply_fn, emission_cov_fn, log_emission_cov, loss_fn, log_init_cov,
        dynamics_weights, memory_size, method, X_eval, y_eval
    ): 
        if emission_cov_fn is None:
            emission_cov_fn = lambda w, x: jnp.power(10, log_emission_cov)
        model_params = RebayesParams(
            initial_mean=params,
            initial_covariance=jnp.power(10, log_init_cov).item(),
            dynamics_weights=dynamics_weights,
            dynamics_covariance=None,
            emission_mean_function=apply_fn,
            emission_cov_function=emission_cov_fn,
        )
        lofi_params = LoFiParams(
            memory_size=memory_size,
        )

        estimator = RebayesLoFi(model_params, lofi_params, method=method)

        # 5-fold cross-validation
        train_val_splits = LoFiManager._cross_validation_splits(5, X_eval, y_eval)

        # train and evaluate
        val_losses = []
        for X_train, y_train, X_val, y_val in train_val_splits:
            bel, _  = estimator.scan(X_train, y_train, progress_bar=False)

            # evaluate
            y_pred = vmap(jit(apply_fn), (None, 0))(bel.mean, X_val)
            val_loss = jnp.mean(loss_fn(y_val, y_pred))
            val_losses.append(val_loss)
        
        val_loss = jnp.mean(jnp.array(val_losses))
        if jnp.isnan(val_loss):
            return -1e3

        return max(-val_loss, -1e3)

    def choose_hparams(self):
        """Tune and return hyper-parameters using Bayesian Optimization."""
        # initialize model
        model_params = RebayesParams(
            initial_mean=self.params,
            initial_covariance=None,
            dynamics_weights=None,
            dynamics_covariance=None,
            emission_mean_function=self.apply_fn,
            emission_cov_function=lambda w, x: None,
        )
        if self.method_title == 'orfit':
            pass
        else:
            lofi_opt_fn = lambda log_init_cov, dynamics_weights, log_emission_cov: LoFiManager._opt_fn(
                params=self.params,
                apply_fn=self.apply_fn,
                emission_cov_fn=self.emission_cov_fn,
                log_emission_cov=log_emission_cov,
                loss_fn=self.loss,
                log_init_cov=log_init_cov,
                dynamics_weights=dynamics_weights,
                memory_size=self.memory,
                method=self.method_title,
                X_eval=self.X_warmup,
                y_eval=self.y_warmup,
            )
            opt_fn = {
                'f': lofi_opt_fn,
                'pbounds': {
                    'log_init_cov': (-7.0, 0.0),
                    'dynamics_weights': (0.0, 1.0),
                    'log_emission_cov': (-7.0, 0.0),
                }
            }
            if self.method_title == 'aov_lofi' or self.dataset[-14:] == 'classification':
                lofi_opt_fn = lambda log_init_cov, dynamics_weights: lofi_opt_fn(log_init_cov, dynamics_weights, None)
                opt_fn = {
                    'f': lofi_opt_fn,
                    'pbounds': {
                        'log_init_cov': (-7.0, 0.0),
                        'dynamics_weights': (0.0, 1.0),
                    }
                }
            optimizer = BayesianOptimization(**opt_fn)
            optimizer.maximize(init_points=5, n_iter=5)
            if self.emission_cov_fn:
                emission_cov_function = self.emission_cov_fn
            else:
                if 'log_emission_cov' in optimizer.max['params']:
                    emission_cov_function = lambda w, x: jnp.power(10, optimizer.max['params']['log_emission_cov'])
                else:
                    emission_cov_function = lambda w, x: None
            model_params = RebayesParams(
                initial_mean=self.params,
                initial_covariance=jnp.power(10, optimizer.max['params']['log_init_cov']).item(),
                dynamics_weights=optimizer.max['params']['dynamics_weights'],
                dynamics_covariance=None,
                emission_mean_function=self.apply_fn,
                emission_cov_function=emission_cov_function,
            )

        lofi_params = LoFiParams(
            memory_size=self.memory,
        )
        estimator = RebayesLoFi(model_params, lofi_params, method=self.method_title)

        return estimator
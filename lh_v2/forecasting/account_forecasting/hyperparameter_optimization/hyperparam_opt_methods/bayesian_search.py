"""
Bayesian Optimization for hyperparameter tuning.

This module implements Bayesian optimization using Optuna's TPE (Tree-structured
Parzen Estimator) algorithm. Unlike grid/random search which sample independently,
Bayesian optimization learns from past trials to intelligently select next parameters.
"""

from __future__ import annotations

import datetime
import time

import numpy as np
import optuna

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params
import lh_v2.params.forecasting_params.account_forecasting_params as afp
from lh_v2.util import get_logger

from ...account_forecasting_methods import AbstractAccountForecastingMethod

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def optimize_hyperparameters_bayesian(
    info: dts.AccountDriverGroup,
    best_lags: dict[dts.DriverName, int],
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    hyperopt_params: params.HyperparamOptParams,
    forecasting_method: type[AbstractAccountForecastingMethod],
    base_params: afp.BaseAccountForecastParams,
    params_range: afp.BaseAccountForecastParamsRange,
    selection_metric: aft.AccountValidationMetricEnum,
) -> afp.BaseAccountForecastParams:
    """
    Perform Bayesian hyperparameter optimization using Optuna.

    Args:
        info: Account and driver data
        best_lags: Optimized lags for each driver
        training_daterange: Training data date range
        forecast_daterange: Validation data date range
        hyperopt_params: Hyperparameter optimization configuration
        forecasting_method: Forecasting method class
        base_params: Base parameters for the method
        params_range: Parameter ranges for optimization
        selection_metric: Metric to optimize

    Returns:
        Optimized parameters
    """
    # Calculate total trials based on n_options_per_param
    # For Bayesian opt, this represents the total budget rather than per-param samples
    n_params = len(type(params_range).model_fields)

    # If no parameters to optimize, return base params
    if n_params == 0:
        logger.info(
            f'No hyperparameters to optimize for {forecasting_method.__name__}. '
            f'Returning base parameters.'
        )
        return base_params

    # Use n_options_per_param as a multiplier to get reasonable trial count
    # e.g., 3 options/param * 7 params * 5 multiplier = 105 trials
    n_trials = (
        hyperopt_params.n_options_per_param
        * n_params
        * hyperopt_params.bayesian_trial_multiplier
    )

    logger.info(
        f'Starting Bayesian hyperparameter optimization for '
        f'{forecasting_method.__name__} with {n_trials} trials '
        f'across {n_params} parameters.'
    )

    # Create method instance
    method_instance = forecasting_method(
        info=info,
        best_lags=best_lags,
        training_daterange=training_daterange,
        forecast_daterange=forecast_daterange,
        model_params=base_params,
    )
    method_instance.set_training_data()
    method_instance.set_forecasting_input_data()
    method_instance.set_validation_data()

    # Track timings
    timings: list[float] = []
    start_time = time.time()

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function to minimize."""
        trial_start = time.time()

        # Sample parameters based on their types and ranges
        params_dict = {}
        for field_name in type(params_range).model_fields.keys():
            param_name = field_name[:-6]  # Remove '_range' suffix
            param_range = getattr(params_range, field_name)

            # Determine if int or float based on the range values
            if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                params_dict[param_name] = trial.suggest_int(
                    param_name, param_range[0], param_range[1]
                )
            else:
                params_dict[param_name] = trial.suggest_float(
                    param_name, param_range[0], param_range[1]
                )

        # Create params object
        params_obj = type(base_params)(**params_dict)

        # Train and validate
        method_instance.set_model_params(model_params=params_obj)
        method_instance.train()
        _, validation_metrics = method_instance.validate()

        timings.append(time.time() - trial_start)

        return validation_metrics[selection_metric]

    # Create study with TPE sampler (Bayesian optimization)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Get best parameters
    best_params_dict = study.best_params
    best_params = type(base_params)(**best_params_dict)

    # Log timing statistics
    timings_array = np.array(timings)
    logger.timing(
        'Bayesian Hyperparameter Optimization Timings:\n'
        f'\tForecasting Method - `{forecasting_method.name()}`\n'
        f'\tHyperparameter Optimization Method - `{hyperopt_params.method.value}`\n'
        f'\tNumber of trials - {n_trials}.\n'
        f'\tTotal time - {time.time() - start_time:.4f} seconds.\n'
        f'\tMean time - {np.mean(timings_array):.4f} seconds.\n'
        f'\tMedian time - {np.median(timings_array):.4f} seconds.\n'
        f'\tMax time - {np.max(timings_array):.4f} seconds.\n'
        f'\tMin time - {np.min(timings_array):.4f} seconds.\n'
        f'\tStandard Deviation - {np.std(timings_array):.4f} seconds.\n'
        f'\tBest {selection_metric.value}: {study.best_value:.4f}\n'
    )

    return best_params

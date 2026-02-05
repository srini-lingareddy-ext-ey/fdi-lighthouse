import datetime
import time

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params
import lh_v2.params.forecasting_params.account_forecasting_params as afp
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from ..account_forecasting_methods import AbstractAccountForecastingMethod
from .hyperparam_opt_methods import (
    INDEPENDENT_HYPEROPT_METHOD_MAP,
    HyperparamOptMethodEnum,
)
from .indepentent_opt import make_param_options

logger = get_logger(__name__)


def optimize_hyperparameters(
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
    Optimize hyperparameters for an account forecasting method.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        Account driver group information containing the data to forecast.
    best_lags : dict[dts.DriverName, int]
        Dictionary mapping driver names to their optimal lag values.
    training_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the training period.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the forecasting period.
    hyperopt_params : params.HyperparamOptParams
        Hyperparameter optimization configuration parameters.
    forecasting_method : type[AbstractAccountForecastingMethod]
        The forecasting method class to optimize.
    base_params : afp.BaseAccountForecastParams
        Base parameters for the forecasting method.
    params_range : afp.BaseAccountForecastParamsRange
        Range of parameter values to search over during optimization.
    selection_metric : aft.AccountValidationMetricEnum
        Validation metric to use for selecting the best parameters.

    Returns
    -------
    afp.BaseAccountForecastParams
        Optimized parameters that minimize the selection metric.

    Raises
    ------
    NotImplementedError
        If a non-independent hyperparameter optimization method is requested.
    ValueError
        If the selection metric is not found in validation metrics.

    Notes
    -----
    If `hyperopt_params.b_optimize` is False, returns the base parameters without
    optimization. Currently only supports independent hyperparameter optimization
    methods.
    """
    # Skip optimization if disabled in configuration
    if not hyperopt_params.b_optimize:
        return base_params

    # Skip optimization for disabled forecasting methods
    if forecasting_method.method_enum() in hyperopt_params.disabled_methods:
        logger.info(
            f'Hyperparameter optimization is disabled for '
            f'{forecasting_method.name()}. Returning base parameters.'
        )
        return base_params

    # Route to appropriate optimizer based on method type
    if INDEPENDENT_HYPEROPT_METHOD_MAP[hyperopt_params.method]:
        return _optimize_independent(
            info=info,
            best_lags=best_lags,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            hyperopt_params=hyperopt_params,
            forecasting_method=forecasting_method,
            base_params=base_params,
            params_range=params_range,
            selection_metric=selection_metric,
        )
    else:
        return _optimize_sequential(
            info=info,
            best_lags=best_lags,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            hyperopt_params=hyperopt_params,
            forecasting_method=forecasting_method,
            base_params=base_params,
            params_range=params_range,
            selection_metric=selection_metric,
        )


def _optimize_independent(
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
    Optimize using independent methods (GRID, RANDOM).

    These methods pre-generate all parameter combinations and can evaluate
    them in any order (potentially in parallel in the future).
    """
    # Pre-generate all parameter combinations
    param_options = make_param_options(
        base_params=base_params,
        params_range=params_range,
        n_options_per_param=hyperopt_params.n_options_per_param,
        method=hyperopt_params.method,
    )
    logger.info(
        f'Starting {hyperopt_params.method.value} hyperparameter optimization for '
        f'{forecasting_method.__name__} with {len(param_options)} parameter combinations '
        f'on account `{info.account.account_type}`.'
    )

    # Initialize the forecasting method instance with base parameters
    method_instance = forecasting_method(
        info=info,
        best_lags=best_lags,
        training_daterange=training_daterange,
        forecast_daterange=forecast_daterange,
        model_params=base_params,
    )
    # Prepare all required data for training and validation
    method_instance.set_training_data()
    method_instance.set_forecasting_input_data()
    method_instance.set_validation_data()

    # Train with base parameters and validate to ensure selection metric exists
    method_instance.train()
    _, validation_metrics = method_instance.validate()
    if selection_metric not in validation_metrics.keys():
        raise ValueError(
            f'Selection metric {selection_metric} not found in validation metrics.'
        )

    # Initialize tracking variables for best parameters and performance metrics
    best_metric_value = float('inf')
    best_params: afp.BaseAccountForecastParams = base_params
    timings: ArrayF = np.zeros(len(param_options))
    start_time = time.time()

    # Iterate through all parameter combinations to find the best one
    for idx, params_option in enumerate(param_options):
        last_time = time.time()

        # Update model with current parameter combination
        method_instance.set_model_params(model_params=params_option)
        method_instance.train()
        _, validation_metrics = method_instance.validate()

        # Track execution time for this parameter combination
        timings[idx] = time.time() - last_time

        # Update best parameters if this combination performs better
        if validation_metrics[selection_metric] < best_metric_value:
            best_metric_value = validation_metrics[selection_metric]
            best_params = params_option

    # Log detailed timing statistics for the optimization process
    logger.timing(
        'Hyperparameter Optimization Timings:\n'
        f'\tForecasting Method - `{forecasting_method.name()}`\n'
        f'\tHyperparameter Optimization Method - `{hyperopt_params.method.value}`\n'
        f'\tNumber of combinations - {len(param_options)}.\n'
        f'\tTotal time - {time.time() - start_time:.6f} seconds.\n'
        f'\tMean time - {np.mean(timings):.6f} seconds.\n'
        f'\tMedian time - {np.median(timings):.6f} seconds.\n'
        f'\tMax time - {np.max(timings):.6f} seconds.\n'
        f'\tMin time - {np.min(timings):.6f} seconds.\n'
        f'\tStandard Deviation - {np.std(timings):.6f} seconds.\n'
    )

    return best_params


def _optimize_sequential(
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
    Optimize using sequential methods (BAYESIAN, etc.).

    These methods generate parameters iteratively, where each new parameter
    combination depends on the results of previous evaluations.
    """
    # Route to specific sequential method
    if hyperopt_params.method == HyperparamOptMethodEnum.BAYESIAN:
        from .hyperparam_opt_methods.bayesian_search import (
            optimize_hyperparameters_bayesian,
        )

        return optimize_hyperparameters_bayesian(
            info=info,
            best_lags=best_lags,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            hyperopt_params=hyperopt_params,
            forecasting_method=forecasting_method,
            base_params=base_params,
            params_range=params_range,
            selection_metric=selection_metric,
        )
    else:
        raise NotImplementedError(
            f'Sequential optimization method {hyperopt_params.method.value} is not implemented. '
            f'Available sequential methods: {HyperparamOptMethodEnum.BAYESIAN.value}'
        )

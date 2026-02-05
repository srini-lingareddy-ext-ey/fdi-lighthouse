import datetime
import time
from typing import Sequence

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params
from lh_v2.util import get_logger

from ...driver_forecasting import create_driver_forecasts
from ...driver_forecasting.driver_forecasting_types import DriverForecastingInput
from ..account_forecasting_methods import (
    ACCOUNT_FORECASTING_METHOD_MAP,
    AbstractAccountForecastingMethod,
)
from ..hyperparameter_optimization import optimize_hyperparameters
from .model_training_types import (
    ModelTrainingInput,
    ModelTrainingOutput,
)
from .train_models_util import select_methods

logger = get_logger(__name__)


def train_method(
    info: dts.AccountDriverGroup,
    best_lags: dict[dts.DriverName, int],
    method: aft.AccountForecastingMethodEnum,
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    method_params: params.forecasting_params.account_forecasting_params.BaseAccountForecastParams,
) -> AbstractAccountForecastingMethod:
    """
    Train a single forecasting method for a given account driver group.

    This function instantiates a specific forecasting method, trains it on the
    provided data using the specified parameters and date ranges, and returns
    the trained model instance.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        The account driver group containing training data, including account
        information and associated drivers.
    best_lags : dict[dts.DriverName, int]
        Dictionary mapping driver names to their optimal lag values.
    method : aft.AccountForecastingMethodEnum
        The forecasting method enum specifying which method to train.
    training_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the training period (inclusive).
    forecast_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the forecast period (inclusive).
    method_params : params.forecasting_params.account_forecasting_params.BaseAccountForecastParams
        Method-specific model parameters for training.

    Returns
    -------
    AbstractAccountForecastingMethod
        The trained forecasting method instance ready for validation or prediction.

    Notes
    -----
    The function looks up the appropriate method class from ACCOUNT_FORECASTING_METHOD_MAP
    using the provided enum, instantiates it with the given parameters, and trains it
    in a single pass.

    See Also
    --------
    train_methods : Trains multiple forecasting methods with hyperparameter optimization.
    ACCOUNT_FORECASTING_METHOD_MAP : Mapping of method enums to their implementation classes.
    """
    logger.info(
        f'Training method `{method.name}` on account `{info.account.account_type}`.'
    )

    # Get the class corresponding to the method enum
    method_class = ACCOUNT_FORECASTING_METHOD_MAP[method]
    # Instantiate the method with training data and parameters
    method_instance = method_class(
        info=info,
        best_lags=best_lags,
        training_daterange=training_daterange,
        forecast_daterange=forecast_daterange,
        model_params=method_params,
    )
    # Train the forecasting method
    method_instance.train()

    return method_instance


def train_methods(
    info: dts.AccountDriverGroup,
    best_lags: dict[dts.DriverName, int],
    selected_methods: Sequence[aft.AccountForecastingMethodEnum],
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    af_params: params.AccountForecastParams,
) -> tuple[
    dict[aft.AccountForecastingMethodEnum, AbstractAccountForecastingMethod],
    params.AccountForecastParams,
]:
    """
    Train multiple forecasting methods with optimized hyperparameters.

    This function iterates over selected forecasting methods, optimizes their
    hyperparameters, trains each method, and returns a dictionary of trained
    model instances along with updated parameters.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        The account driver group containing training data, including account
        information and associated drivers.
    best_lags : dict[dts.DriverName, int]
        Dictionary mapping driver names to their optimal lag values.
    selected_methods : Sequence[aft.AccountForecastingMethodEnum]
        Sequence of forecasting method enums to train.
    training_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the training period (inclusive).
    forecast_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the forecast period (inclusive).
    af_params : params.AccountForecastParams
        Account forecasting parameters, including base parameters for each
        method and hyperparameter optimization settings.

    Returns
    -------
    trained_methods : dict[aft.AccountForecastingMethodEnum, AbstractAccountForecastingMethod]
        Dictionary mapping each selected method enum to its trained method
        instance with optimized hyperparameters.
    updated_af_params : params.AccountForecastParams
        Updated AccountForecastParams with optimized hyperparameters for each method.

    Notes
    -----
    For each method, the function:
    1. Optimizes hyperparameters using the specified optimization parameters
       and MAPE as the selection metric
    2. Instantiates the method class with optimized parameters
    3. Trains the method on the provided data
    4. Stores the trained instance and updates parameters
    5. Logs training time for performance monitoring

    See Also
    --------
    optimize_hyperparameters : Function that performs hyperparameter optimization.
    train_method : Trains a single method without hyperparameter optimization.
    """

    # Initialize dictionary to hold trained method instances
    trained_methods: dict[
        aft.AccountForecastingMethodEnum, AbstractAccountForecastingMethod
    ] = {}

    updated_af_params = af_params.model_copy()

    # Iterate over each selected forecasting method and train with optimized hyperparameters
    for method in selected_methods:
        # Optimize hyperparameters for the current method using the training data
        # This searches over the parameter range to find the best configuration based on MAPE
        method_params = optimize_hyperparameters(
            info=info,
            best_lags=best_lags,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            hyperopt_params=af_params.hyperparam_opt_params,
            forecasting_method=ACCOUNT_FORECASTING_METHOD_MAP[method],
            base_params=af_params[method],
            params_range=af_params.hyperparam_opt_params[method],
            selection_metric=aft.AccountValidationMetricEnum.MAPE,
        )

        logger.info(
            f'Training method `{method.name}` on account `{info.account.account_type}`.'
        )

        last_time = time.time()

        # Get the forecasting method class corresponding to the method enum
        method_class = ACCOUNT_FORECASTING_METHOD_MAP[method]

        # Instantiate the method with account data, lags, date ranges, and optimized parameters
        method_instance = method_class(
            info=info,
            best_lags=best_lags,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            model_params=method_params,
        )

        # Train the forecasting method on the provided data
        method_instance.train()

        # Store the trained instance in the dictionary for later validation
        trained_methods[method] = method_instance

        updated_af_params.set_method_params(method, method_params)

        logger.timing(
            f'Training method `{method_instance.name()}` - {time.time() - last_time:.6f} seconds.'
        )

    # Return the dictionary of trained methods, and an
    # updated AccountForecastParams with optimized parameters
    return trained_methods, updated_af_params


def _validate_methods(
    selected_methods: Sequence[aft.AccountForecastingMethodEnum],
    trained_methods: dict[
        aft.AccountForecastingMethodEnum, AbstractAccountForecastingMethod
    ],
) -> tuple[list[dts.AccountInfo], list[dict[aft.AccountValidationMetricEnum, float]]]:
    """
    Validate trained forecasting methods and collect performance metrics.

    This function iterates through trained forecasting methods, validates each one,
    and collects both the forecasted accounts and their performance metrics.

    Parameters
    ----------
    selected_methods : Sequence[aft.AccountForecastingMethodEnum]
        Sequence of forecasting method enums in the order to validate.
    trained_methods : dict[aft.AccountForecastingMethodEnum, AbstractAccountForecastingMethod]
        Dictionary mapping method enums to their trained method instances.

    Returns
    -------
    forecasted_accounts : list[dts.AccountInfo]
        List of forecasted account information, one per validated method, in the
        same order as selected_methods.
    validation_metrics : list[dict[aft.AccountValidationMetricEnum, float]]
        List of dictionaries containing validation metrics for each method, in the
        same order as selected_methods. Each dictionary maps metric enums to their
        computed float values.

    Notes
    -----
    The function calls the validate() method on each trained model instance, which
    computes forecasts and evaluates them against validation data. Validation time
    for each method is logged for performance monitoring.

    See Also
    --------
    train_methods : Function that creates the trained_methods dictionary.
    AbstractAccountForecastingMethod.validate : Method that performs validation.
    """

    forecasted_accounts: list[dts.AccountInfo] = []
    # Initialize dictionary to hold validation metrics for each method
    validation_metrics: list[dict[aft.AccountValidationMetricEnum, float]] = []

    # Iterate over each trained method and validate using the provided data
    for method in selected_methods:
        logger.info(
            f'Validating method `{trained_methods[method].name()}` '
            f'on account `{trained_methods[method].info.account.account_type}`.'
        )
        last_time = time.time()
        # Call the validate method of the trained instance
        forecasted_account, metric = trained_methods[method].validate()
        # Store the resulting metrics in the dictionary
        forecasted_accounts.append(forecasted_account)
        validation_metrics.append(metric)
        logger.timing(
            f'Validating method `{trained_methods[method].name()}` '
            f'- {time.time() - last_time:.6f} seconds.'
        )

    # Return the dictionary of validation metrics
    return forecasted_accounts, validation_metrics


def train_and_validate_models(
    model_training_info: ModelTrainingInput,
    general_params: params.GeneralParams,
    af_params: params.AccountForecastParams
    | dict[dts.AccountType, params.AccountForecastParams],
    df_params: params.DriverForecastParams,
) -> ModelTrainingOutput:
    """
    Train and validate account forecasting models across multiple accounts and methods.

    This function orchestrates the complete model training and validation pipeline,
    including driver forecasting, model training with hyperparameter optimization,
    and validation. It processes each account independently and returns consolidated
    results for all accounts and methods.

    Parameters
    ----------
    model_training_info : ModelTrainingInput
        Input containing account-driver groups, optimal lags, and driver classifications
        for each account.
    general_params : params.GeneralParams
        General parameters including training and validation date ranges.
    af_params : params.AccountForecastParams | dict[dts.AccountType, params.AccountForecastParams]
        Account forecasting parameters. Can be either:
        - A single AccountForecastParams applied to all accounts
        - A dictionary mapping account types to account-specific parameters
    df_params : params.DriverForecastParams
        Driver forecasting parameters used to generate driver forecasts for the
        validation period.

    Returns
    -------
    ModelTrainingOutput
        Comprehensive training output containing:
        - account_map: Dictionary mapping account types to their indices
        - method_map: Nested dictionary mapping account types to method enums to indices
        - forecasted_accounts: Nested list of forecasted accounts [account_idx][method_idx]
        - metrics: Nested list of validation metrics [account_idx][method_idx]
        - best_params: List of updated AccountForecastParams with optimized hyperparameters

    Notes
    -----
    The function processes each account sequentially through these steps:
    1. Extracts account-specific information and optimal lags
    2. Generates driver forecasts for the validation period
    3. Trains all selected forecasting methods with hyperparameter optimization
    4. Validates each method and collects forecasts and performance metrics
    5. Stores results in structured format for easy access

    The function supports both uniform parameters across all accounts and
    account-specific parameter configurations.

    See Also
    --------
    train_methods : Function that trains multiple methods for one account.
    create_driver_forecasts : Function that generates driver forecasts.
    _validate_methods : Function that validates trained methods.
    ModelTrainingOutput : Output data structure for training results.

    Examples
    --------
    >>> output = train_and_validate_models(
    ...     model_training_info=training_input,
    ...     general_params=gen_params,
    ...     af_params=account_forecast_params,
    ...     df_params=driver_forecast_params
    ... )
    >>> # Access results for a specific account
    >>> revenue_forecasts = output.forecasted_accounts[output.account_map[AccountType('REVENUE')]]
    >>> revenue_metrics = output.metrics[output.account_map[AccountType('REVENUE')]]
    """

    # Determine which forecasting methods to use based on account forecast parameters
    selected_methods: (
        list[aft.AccountForecastingMethodEnum]
        | dict[dts.AccountType, list[aft.AccountForecastingMethodEnum]]
    )
    if isinstance(af_params, params.AccountForecastParams):
        selected_methods = select_methods(method_params=af_params.methods)
    else:
        selected_methods = {}
        for (
            account
        ) in model_training_info.accounts_drivers.accounts.get_ordered_accounts():
            selected_methods[account] = select_methods(
                method_params=af_params[account].methods
            )

    # Initialize containers to store validation metrics and forecasts for all accounts
    metrics: list[list[dict[aft.AccountValidationMetricEnum, float]]] = []
    forecasts: list[list[dts.AccountInfo]] = []
    updated_af_params_lst: list[params.AccountForecastParams] = []
    method_map: dict[dts.AccountType, dict[aft.AccountForecastingMethodEnum, int]] = {}

    # Process each account in the training data
    for account in model_training_info.accounts_drivers.accounts.get_ordered_accounts():
        # Extract account-specific information and optimal lags
        info = model_training_info.accounts_drivers[account]
        best_lags = model_training_info.lags[account]

        # Prepare input for driver forecasting using training and validation periods
        df_input = DriverForecastingInput(
            drivers=info.drivers,
            lags=best_lags,
            training_daterange=(
                general_params.training_start_date,
                general_params.training_end_date,
            ),
            forecast_daterange=(
                general_params.validation_start_date,
                general_params.validation_end_date,
            ),
        )

        # Generate driver forecasts for the validation period
        forecasted_info = dts.AccountDriverGroup(
            account=info.account,
            drivers=create_driver_forecasts(
                forecasting_info=df_input,
                df_params=df_params,
            ),
            np_dtype=info.np_dtype,
        )

        # Train all selected forecasting methods using the forecasted driver data
        if isinstance(af_params, params.AccountForecastParams):
            selected_af_params = af_params
        else:
            selected_af_params = af_params[account]

        if not isinstance(selected_methods, dict):
            selected_methods_account = selected_methods
        else:
            selected_methods_account = selected_methods[account]

        trained_methods, updated_af_params = train_methods(
            info=forecasted_info,
            best_lags=best_lags,
            selected_methods=selected_methods_account,
            training_daterange=(
                general_params.training_start_date,
                general_params.training_end_date,
            ),
            forecast_daterange=(
                general_params.validation_start_date,
                general_params.validation_end_date,
            ),
            af_params=selected_af_params,
        )

        # Validate each trained method and collect forecasts and performance metrics
        forecasted_accounts, validation_metrics = _validate_methods(
            selected_methods=selected_methods_account,
            trained_methods=trained_methods,
        )

        # Store results for this account
        forecasts.append(forecasted_accounts)
        metrics.append(validation_metrics)
        updated_af_params_lst.append(updated_af_params)

        # Create a mapping from method enums to their corresponding indices
        method_map[account] = {
            method: idx for idx, method in enumerate(selected_methods_account)
        }

    # Return consolidated training and validation results
    return ModelTrainingOutput(
        account_map=model_training_info.accounts_drivers.accounts.account_map,
        method_map=method_map,
        forecasted_accounts=forecasts,
        metrics=metrics,
        best_params=updated_af_params_lst,
    )

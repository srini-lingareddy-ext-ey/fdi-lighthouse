import pathlib as pth
from typing import Any, Optional

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.account_reconciliation import apply_account_reconciliation
from lh_v2.forecasting import create_account_forecasts, train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    AccountForecastingMethodEnum,
    ModelForecastingInput,
    ModelForecastingOutput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.shared import ArrayF

from .new_instance_util import handle_driver_analysis, initialize


def forecast_models_new_llm(
    config_info: pth.Path | dict[str, Any],
    account_data_pth: pth.Path,
    driver_data_pth: pth.Path,
    selected_drivers: Optional[
        dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]]
    ] = None,
    drivers_lags: Optional[dict[dts.AccountType, dict[dts.DriverName, int]]] = None,
    selected_models: Optional[
        dict[dts.AccountType, AccountForecastingMethodEnum]
    ] = None,
    model_params: Optional[dict[dts.AccountType, params.AccountForecastParams]] = None,
    validation_errors: Optional[dict[dts.AccountType, ArrayF]] = None,
) -> ModelForecastingOutput:
    """
    Generate forecasts for new Lighthouse instances with optional LLM-selected configurations.

    This function handles the complete forecasting workflow, including parameter initialization,
    data loading, driver analysis, model training (if needed), and forecast generation.
    It supports three modes: (1) full automation with driver and model selection, (2) LLM-selected
    models with automatic training, or (3) fully specified models with pre-trained parameters.

    Parameters
    ----------
    config_info : pathlib.Path or dict[str, Any]
        Path to YAML configuration file or dictionary of configuration parameters
        containing general, driver analysis, account forecast, and driver forecast parameters.
    account_data_pth : pathlib.Path
        Path to the account data source file.
    driver_data_pth : pathlib.Path
        Path to the driver data source file.
    selected_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]], optional
        Pre-selected drivers organized by account type and classification.
        If None, automatic driver analysis will be performed. Default is None.
    drivers_lags : dict[AccountType, dict[DriverName, int]], optional
        Lag values for each driver by account type.
        If None, optimal lags will be determined through analysis. Default is None.
    selected_models : dict[AccountType, AccountForecastingMethodEnum], optional
        Pre-selected forecasting models for each account type.
        If None, models will be automatically selected based on validation performance. Default is None.
    model_params : dict[AccountType, AccountForecastParams], optional
        Pre-trained model parameters for each account type.
        If None and selected_models is provided, training will be performed. Default is None.
        Must be provided together with validation_errors if selected_models is provided.
    validation_errors : dict[AccountType, ArrayF], optional
        Validation errors for the selected models by account type.
        If None and selected_models is provided, validation will be performed. Default is None.
        Must be provided together with model_params if selected_models is provided.

    Returns
    -------
    forecasting_output : ModelForecastingOutput
        Generated forecasts for all account types, including point forecasts,
        prediction intervals, and forecast metadata.

    Notes
    -----
    The function operates in three distinct modes based on provided parameters:

    1. Full automation: If selected_models, model_params, and validation_errors are all None,
       the function performs complete driver analysis, model training, selection, and forecasting.

    2. LLM-selected models: If selected_models is provided but model_params and validation_errors
       are None, the function trains only the selected models and generates forecasts.

    3. Fully specified: If selected_models, model_params, and validation_errors are all provided,
       the function uses the pre-trained models directly for forecasting without additional training.
    """
    # Initialize parameters and load data
    # Parse configuration and load account/driver data into structured format
    lh_params, dr_data = initialize(
        config_info=config_info,
        account_data_pth=account_data_pth,
        driver_data_pth=driver_data_pth,
    )

    # Handle driver analysis to get selected drivers, lags, and classifications
    # If selected_drivers or drivers_lags are None, performs full analysis
    # Otherwise, uses provided selections and derives classifications
    selected_drivers, drivers_lags, drivers_classifications = handle_driver_analysis(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
        selected_drivers=selected_drivers,
        drivers_lags=drivers_lags,
    )

    # Prepare model training input data
    # Filter the driver data to only include the selected drivers
    selected_data = dr_data.select_drivers(selected_drivers=selected_drivers)
    # Package the selected data with lags and classifications into training input
    training_input = ModelTrainingInput(
        accounts_drivers=selected_data,
        lags=drivers_lags,
        classifications=drivers_classifications,
    )

    # Mode 1: Fully specified - use pre-trained models and validation errors directly
    if (
        selected_models is not None
        and model_params is not None
        and validation_errors is not None
    ):
        # Create forecasting input with all pre-specified components
        # No training needed, proceed directly to forecasting
        forecasting_input = ModelForecastingInput(
            accounts_drivers=selected_data,
            lags=drivers_lags,
            classifications=drivers_classifications,
            selected_model=selected_models,
            best_params=model_params,
            validation_errors=validation_errors,
        )
    # Mode 2: LLM-selected models - train only the specified models
    elif (
        selected_models is not None
        and model_params is None
        and validation_errors is None
    ):
        # Configure training parameters to only train the LLM-selected models
        af_params_training: dict[dts.AccountType, params.AccountForecastParams] = {}
        for account in selected_models.keys():
            # Create a copy of the account forecast parameters for this account
            af_params_training[account] = lh_params.account_forecast_params.model_copy()
            # Disable automatic model removal based on performance
            af_params_training[account].methods.b_remove = False
            # Enable manual model selection mode
            af_params_training[account].methods.b_selected = True
            # Specify only the LLM-selected model for training
            af_params_training[account].methods.methods_selected = [
                selected_models[account]
            ]

        # Train and validate only the selected models
        training_results = train_and_validate_models(
            model_training_info=training_input,
            general_params=lh_params.general_params,
            af_params=af_params_training,
            df_params=lh_params.driver_forecast_params,
        )

        # Create forecasting input using training results
        # Extract best parameters and compute validation errors for selected models
        forecasting_input = ModelForecastingInput(
            accounts_drivers=selected_data,
            lags=drivers_lags,
            classifications=drivers_classifications,
            selected_model=selected_models,
            best_params=training_results.format_best_params(),
            validation_errors=training_results.get_selected_methods_errors(
                selected_methods=selected_models,
                actuals=dr_data.accounts,
                val_date_range=(
                    lh_params.general_params.validation_start_date,
                    lh_params.general_params.validation_end_date,
                ),
            ),
        )

    # Mode 3: Full automation - train all models and select best performers
    else:
        # Train and validate all available models using full parameter set
        training_results = train_and_validate_models(
            model_training_info=training_input,
            general_params=lh_params.general_params,
            af_params=lh_params.account_forecast_params,
            df_params=lh_params.driver_forecast_params,
        )

        # Create forecasting input with automatically selected best models
        # Select models based on validation performance
        forecasting_input = ModelForecastingInput(
            accounts_drivers=selected_data,
            lags=drivers_lags,
            classifications=drivers_classifications,
            selected_model=training_results.select_forecast_methods(),
            best_params=training_results.format_best_params(),
            validation_errors=training_results.get_selected_methods_errors(
                selected_methods=training_results.select_forecast_methods(),
                actuals=dr_data.accounts,
                val_date_range=(
                    lh_params.general_params.validation_start_date,
                    lh_params.general_params.validation_end_date,
                ),
            ),
        )

    # Create forecasts for multiple accounts
    # Generate point forecasts and prediction intervals using selected models
    forecast_output = create_account_forecasts(
        forecasting_input=forecasting_input,
        general_params=lh_params.general_params,
        df_params=lh_params.driver_forecast_params,
    )

    # Apply account reconciliation if enabled in general parameters
    return apply_account_reconciliation(
        forecasting_data=forecast_output,
        reconciliation_params=lh_params.account_reconciliation_params,
    )

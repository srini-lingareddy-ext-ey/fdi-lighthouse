import pathlib as pth
from typing import Any, Optional

import lh_v2.datatypes as dts
from lh_v2.forecasting import train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
    ModelTrainingOutput,
)

from .new_instance_util import handle_driver_analysis, initialize


def train_models_new_llm(
    config_info: pth.Path | dict[str, Any],
    account_data_pth: pth.Path,
    driver_data_pth: pth.Path,
    selected_drivers: Optional[
        dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]]
    ] = None,
    drivers_lags: Optional[dict[dts.AccountType, dict[dts.DriverName, int]]] = None,
) -> ModelTrainingOutput:
    """
    Train and validate forecasting models for new Lighthouse instances with LLM-selected drivers.

    This function handles the complete workflow for model training, including parameter
    initialization, data loading, driver analysis (if needed), and model training/validation.
    It can either perform automatic driver selection or use pre-selected drivers from LLM output.

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
        Typically provided from LLM-based driver selection results.
    drivers_lags : dict[AccountType, dict[DriverName, int]], optional
        Lag values for each driver by account type.
        If None, optimal lags will be determined through analysis. Default is None.
        Typically provided from LLM-based driver selection results.

    Returns
    -------
    training_output : ModelTrainingOutput
        Results of model training and validation, including trained models,
        validation metrics, and performance statistics for each account type.
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
        output_params=lh_params.output_params,
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

    # Train and validate models using the selected drivers and lags
    # This performs the full model training pipeline including cross-validation
    return train_and_validate_models(
        model_training_info=training_input,
        general_params=lh_params.general_params,
        af_params=lh_params.account_forecast_params,
        df_params=lh_params.driver_forecast_params,
        output_params=lh_params.output_params,
    )

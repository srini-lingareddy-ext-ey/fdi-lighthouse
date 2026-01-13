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
    # Initialize parameters and load data
    lh_params, dr_data = initialize(
        config_info=config_info,
        account_data_pth=account_data_pth,
        driver_data_pth=driver_data_pth,
    )

    # Handle driver analysis to get selected drivers, lags, and classifications
    selected_drivers, drivers_lags, drivers_classifications = handle_driver_analysis(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
        selected_drivers=selected_drivers,
        drivers_lags=drivers_lags,
    )

    # Prepare model training input data
    selected_data = dr_data.select_drivers(selected_drivers=selected_drivers)
    training_input = ModelTrainingInput(
        accounts_drivers=selected_data,
        lags=drivers_lags,
        classifications=drivers_classifications,
    )

    # Train and validate models using the selected drivers and lags
    return train_and_validate_models(
        model_training_info=training_input,
        general_params=lh_params.general_params,
        af_params=lh_params.account_forecast_params,
        df_params=lh_params.driver_forecast_params,
    )

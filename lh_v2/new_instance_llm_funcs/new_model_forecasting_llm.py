import pathlib as pth
from typing import Any, Optional

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.forecasting import create_account_forecasts, train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    AccountForecastingMethodEnum,
    ModelForecastingInput,
    ModelForecastingOutput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)

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
) -> ModelForecastingOutput:
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

    if selected_models is not None and model_params is not None:
        forecasting_input = ModelForecastingInput(
            accounts_drivers=selected_data,
            lags=drivers_lags,
            classifications=drivers_classifications,
            selected_model=selected_models,
            best_params=model_params,
        )
    elif selected_models is not None and model_params is None:
        af_params_training: dict[dts.AccountType, params.AccountForecastParams] = {}
        for account in selected_models.keys():
            af_params_training[account] = lh_params.account_forecast_params.model_copy()
            af_params_training[account].methods.b_remove = False
            af_params_training[account].methods.b_selected = True
            af_params_training[account].methods.methods_selected = [
                selected_models[account]
            ]

        # Train and validate models using the selected drivers and lags
        training_results = train_and_validate_models(
            model_training_info=training_input,
            general_params=lh_params.general_params,
            af_params=af_params_training,
            df_params=lh_params.driver_forecast_params,
        )

        forecasting_input = ModelForecastingInput(
            accounts_drivers=selected_data,
            lags=drivers_lags,
            classifications=drivers_classifications,
            selected_model=selected_models,
            best_params=training_results.format_best_params(),
        )

    else:
        # Train and validate models using the selected drivers and lags
        training_results = train_and_validate_models(
            model_training_info=training_input,
            general_params=lh_params.general_params,
            af_params=lh_params.account_forecast_params,
            df_params=lh_params.driver_forecast_params,
        )

        forecasting_input = ModelForecastingInput(
            accounts_drivers=selected_data,
            lags=drivers_lags,
            classifications=drivers_classifications,
            selected_model=training_results.select_forecast_methods(),
            best_params=training_results.format_best_params(),
        )

    # Create forecasts for multiple accounts
    return create_account_forecasts(
        forecasting_input=forecasting_input,
        general_params=lh_params.general_params,
    )

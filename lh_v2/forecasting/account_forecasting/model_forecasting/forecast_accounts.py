import datetime
import time

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params
from lh_v2.forecasting.driver_forecasting import (
    DriverForecastingInput,
    create_driver_forecasts,
)
from lh_v2.io.output import save_driver_forecasts, save_model_forecast_forecasts
from lh_v2.util import get_logger

from ..account_forecasting_methods import ACCOUNT_FORECASTING_METHOD_MAP
from .model_forecasting_types import ModelForecastingInput, ModelForecastingOutput

logger = get_logger(__name__)


def create_account_forecast(
    info: dts.AccountDriverGroup,
    driver_forecasts: dts.DriverGroup,
    best_lags: dict[dts.DriverName, int],
    selected_method: aft.AccountForecastingMethodEnum,
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    af_params: params.AccountForecastParams,
) -> dts.AccountInfo:
    """
    Create a forecast for a single account using the specified forecasting method.

    This function generates driver forecasts, instantiates the selected forecasting method,
    trains the model on historical data, and produces account-level forecasts for the
    specified forecast period.

    Parameters
    ----------
    info : AccountDriverGroup
        Account and associated driver data for forecasting.
    driver_forecasts : DriverGroup
        Forecasted driver information to be used as input features for account forecasting.
    best_lags : dict[DriverName, int]
        Optimal lag values for each driver associated with this account.
    selected_method : AccountForecastingMethodEnum
        The forecasting method to use (e.g., ARIMAX, Linear Regression, Random Forest).
    training_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the training period, as (start_date, end_date).
    forecast_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the forecast period, as (start_date, end_date).
    af_params : AccountForecastParams
        Account forecasting parameters specific to the selected method.

    Returns
    -------
    account_forecast : AccountInfo
        Forecasted account information including predicted values and metadata.
    """
    logger.info(
        f'Forecasting account `{info.account.account_type}` '
        f'with method `{selected_method.name}`.'
    )

    # Instantiate the forecasting method with account data and configuration
    method_instance = ACCOUNT_FORECASTING_METHOD_MAP[selected_method](
        info=dts.AccountDriverGroup(
            account=info.account,
            drivers=driver_forecasts,
        ),
        best_lags=best_lags,
        training_daterange=training_daterange,
        forecast_daterange=forecast_daterange,
        model_params=af_params[selected_method],
    )

    # Train the model on historical data
    method_instance.train()

    # Generate and return the forecast
    return method_instance.forecast()


def create_account_forecasts(
    forecasting_input: ModelForecastingInput,
    general_params: params.GeneralParams,
    df_params: params.DriverForecastParams,
    output_params: params.OutputParams,
) -> ModelForecastingOutput:
    """
    Create forecasts for multiple accounts using their respective selected methods.

    This function iterates through all accounts in the forecasting input, generates
    individual forecasts using account-specific methods and parameters, and aggregates
    the results into a comprehensive output structure.

    Parameters
    ----------
    forecasting_input : ModelForecastingInput
        Input data containing account-driver information, selected models, optimal lags,
        best parameters, and validation errors for each account.
    general_params : GeneralParams
        General configuration parameters including training, validation, and testing
        date ranges.
    df_params : DriverForecastParams
        Parameters for driver forecasting used across all accounts.

    Returns
    -------
    forecasting_output : ModelForecastingOutput
        Aggregated forecasting results containing forecasts for all accounts,
        the forecast date range, and validation errors.

    Notes
    -----
    The training period for each forecast combines both the training and validation
    periods from general_params to maximize the amount of historical data used for
    final model training before generating forecasts for the testing period.
    """
    # Initialize list to store individual account forecasts
    account_forecasts: list[dts.AccountInfo] = []

    driver_forecasts_all: dict[dts.AccountType, dts.DriverGroup] = {}

    # Iterate through each account in the ordered account list
    for account in forecasting_input.accounts_drivers.accounts.get_ordered_accounts():
        last_time = time.time()
        # Extract account-specific driver information
        info = forecasting_input.accounts_drivers[account]

        driver_forecasts = create_driver_forecasts(
            forecasting_info=DriverForecastingInput(
                drivers=info.drivers,
                lags=forecasting_input.lags[account],
                training_daterange=general_params.get_train_val_daterange(),
                forecast_daterange=general_params.get_testing_daterange(),
            ),
            df_params=df_params,
        )
        driver_forecasts_all[account] = driver_forecasts

        # Generate forecast for current account and append to results
        account_forecasts.append(
            create_account_forecast(
                info=info,
                driver_forecasts=driver_forecasts,
                best_lags=forecasting_input.lags[
                    account
                ],  # Account-specific optimal lags
                selected_method=forecasting_input.selected_model[
                    account
                ],  # Account-specific method
                training_daterange=(
                    general_params.training_start_date,
                    general_params.validation_end_date,
                ),  # Combined training and validation period
                forecast_daterange=(
                    general_params.testing_start_date,
                    general_params.testing_end_date,
                ),  # Testing period for forecast generation
                af_params=forecasting_input.best_params[account],
            )
        )

        logger.timing(
            f'Forecasting account `{account}` using method '
            f'`{forecasting_input.selected_model[account].value}`'
            f' - {time.time() - last_time:.6f} seconds.'
        )

    forecasting_output = ModelForecastingOutput(
        accounts_forecasts=dts.AccountGroupInfo.from_account_lst(account_forecasts),
        forecast_daterange=(
            general_params.testing_start_date,
            general_params.testing_end_date,
        ),
        validation_errors=forecasting_input.validation_errors,
    )

    save_driver_forecasts(
        driver_forecasts=driver_forecasts_all,
        driver_lags=forecasting_input.lags,
        classification_map=forecasting_input.classifications,
        forecasting_daterange=general_params.get_testing_daterange(),
        forecasting_type='forecast',
        output_params=output_params,
    )

    save_model_forecast_forecasts(
        account_forecasts=forecasting_output.accounts_forecasts,
        selected_methods=forecasting_input.selected_model,
        forecast_daterange=general_params.get_testing_daterange(),
        output_params=output_params,
    )

    # Package all forecasts into output structure with forecast date range
    return forecasting_output

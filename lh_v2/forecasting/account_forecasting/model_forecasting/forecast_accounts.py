import datetime
import time

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.params as params
from lh_v2.util import get_logger

from ..account_forecasting_methods import ACCOUNT_FORECASTING_METHOD_MAP
from .model_forecasting_types import ModelForecastingInput, ModelForecastingOutput

logger = get_logger(__name__)


def create_account_forecast(
    info: dts.AccountDriverGroup,
    best_lags: dict[dts.DriverName, int],
    selected_method: aft.AccountForecastingMethodEnum,
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    af_params: params.AccountForecastParams,
) -> dts.AccountInfo:
    """
    Create a forecast for a single account using the specified forecasting method.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        Account information including historical data and associated drivers.
    best_lags : dict[dts.DriverName, int]
        Dictionary mapping driver names to their optimal lag values.
    selected_method : aft.AccountForecastingMethodEnum
        The forecasting method to use for this account.
    training_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the training period.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Start and end dates for the forecast period.
    af_params : params.AccountForecastParams
        Configuration parameters for the account forecasting methods.

    Returns
    -------
    dts.AccountInfo
        The forecasted account information for the specified forecast period.
    """
    # Instantiate the forecasting method with account data and configuration
    method_instance = ACCOUNT_FORECASTING_METHOD_MAP[selected_method](
        info=info,
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
) -> ModelForecastingOutput:
    """
    Create forecasts for multiple accounts using their respective forecasting methods.

    Parameters
    ----------
    forecasting_input : ModelForecastingInput
        Input containing account-driver data, optimal lags, and selected forecasting
        methods for each account.
    general_params : params.GeneralParams
        General configuration parameters including training, validation, and testing
        date ranges.

    Returns
    -------
    ModelForecastingOutput
        Output containing forecasted account information for all accounts and the
        forecast date range.
    """
    # Initialize list to store individual account forecasts
    account_forecasts: list[dts.AccountInfo] = []

    # Iterate through each account in the ordered account list
    for account in forecasting_input.accounts_drivers.accounts.get_ordered_accounts():
        last_time = time.time()
        # Extract account-specific driver information
        info = forecasting_input.accounts_drivers[account]

        # Generate forecast for current account and append to results
        account_forecasts.append(
            create_account_forecast(
                info=info,
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

    # Package all forecasts into output structure with forecast date range
    return ModelForecastingOutput(
        accounts_forecasts=dts.AccountGroupInfo.from_account_lst(account_forecasts),
        forecast_daterange=(
            general_params.testing_start_date,
            general_params.testing_end_date,
        ),
    )

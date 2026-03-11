import datetime
from typing import Sequence

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.data_analysis.account_plotting_util import plot_account_forecasts
from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    AccountForecastingMethodEnum,
)
from lh_v2.datatypes.forecasting_types.driver_forecasting_types import (
    DriverForecastingMethodEnum,
)
from lh_v2.forecasting.account_forecasting.model_forecasting.forecast_accounts import (
    create_account_forecast,
)
from lh_v2.forecasting.driver_forecasting import (
    DriverForecastingInput,
    create_driver_forecasts,
)


def forecast_account_test_driver_forecaster(
    info: dts.AccountDriverGroup,
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    account_forecasting_method: AccountForecastingMethodEnum,
    af_params: params.AccountForecastParams,
    df_params: params.DriverForecastParams,
    driver_forecasting_methods: Sequence[DriverForecastingMethodEnum] | None = None,
    best_lags: dict[dts.DriverName, int] | None = None,
):
    if driver_forecasting_methods is None:
        driver_forecasting_methods = list(DriverForecastingMethodEnum)

    if best_lags is None:
        best_lags = {driver: 0 for driver in info.drivers.get_ordered_drivers()}

    df_params_copy = df_params.model_copy(deep=True)

    account_forecasts: dict[DriverForecastingMethodEnum, dts.AccountInfo] = {}

    for driver_method in driver_forecasting_methods:
        df_params_copy.selected_method = driver_method

        driver_forecasts = create_driver_forecasts(
            forecasting_info=DriverForecastingInput(
                drivers=info.drivers,
                lags=best_lags,
                training_daterange=training_daterange,
                forecast_daterange=forecast_daterange,
            ),
            df_params=df_params_copy,
        )

        account_forecasts[driver_method] = create_account_forecast(
            info=info,
            driver_forecasts=driver_forecasts,
            best_lags=best_lags,
            selected_method=account_forecasting_method,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            af_params=af_params,
        )

    plot_account_forecasts(
        accounts=account_forecasts,
        forecast_daterange=forecast_daterange,
        historicals=info.account,
    )

    return


def forecast_account_test_account_forecasting_methods(
    info: dts.AccountDriverGroup,
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    af_params: params.AccountForecastParams,
    df_params: params.DriverForecastParams,
    account_forecasting_methods: Sequence[AccountForecastingMethodEnum] | None = None,
    driver_forecasting_method: DriverForecastingMethodEnum | None = None,
    best_lags: dict[dts.DriverName, int] | None = None,
):
    if account_forecasting_methods is None:
        account_forecasting_methods = list(AccountForecastingMethodEnum)

    if driver_forecasting_method is None:
        driver_forecasting_method = DriverForecastingMethodEnum.XGBOOST
    df_params_copy = df_params.model_copy(deep=True)
    df_params_copy.selected_method = driver_forecasting_method

    if best_lags is None:
        best_lags = {driver: 0 for driver in info.drivers.get_ordered_drivers()}

    account_forecasts: dict[AccountForecastingMethodEnum, dts.AccountInfo] = {}

    for account_method in account_forecasting_methods:
        driver_forecasts = create_driver_forecasts(
            forecasting_info=DriverForecastingInput(
                drivers=info.drivers,
                lags=best_lags,
                training_daterange=training_daterange,
                forecast_daterange=forecast_daterange,
            ),
            df_params=df_params_copy,
        )

        account_forecasts[account_method] = create_account_forecast(
            info=info,
            driver_forecasts=driver_forecasts,
            best_lags=best_lags,
            selected_method=account_method,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            af_params=af_params,
        )

    plot_account_forecasts(
        accounts=account_forecasts,
        forecast_daterange=forecast_daterange,
        historicals=info.account,
    )

    return

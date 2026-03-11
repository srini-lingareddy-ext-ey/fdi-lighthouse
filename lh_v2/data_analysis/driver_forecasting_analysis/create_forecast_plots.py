import datetime

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.data_analysis.driver_plotting_util import (
    plot_driver_forecast_methods,
)
from lh_v2.datatypes.forecasting_types.driver_forecasting_types import (
    DriverForecastingMethodEnum,
)
from lh_v2.forecasting.driver_forecasting.forecast_drivers import (
    create_singe_driver_forecast,
)


def forecast_driver_methods(
    driver: dts.Driver,
    df_params: params.DriverForecastParams,
    training_daterange: datetime.date | tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    n_lag: int | None = None,
    max_lag: int | None = None,
    methods: list[DriverForecastingMethodEnum] | None = None,
):
    if methods is None:
        methods = list(DriverForecastingMethodEnum)

    if n_lag is None:
        n_lag = 0

    if max_lag is None:
        max_lag = n_lag

    forecasts: dict[DriverForecastingMethodEnum, dts.Driver] = {}

    for method in methods:
        forecasts[method] = create_singe_driver_forecast(
            driver_info=driver,
            df_params=df_params,
            n_lag=n_lag,
            max_lag=max_lag,
            selected_method=method,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
        )

    plot_driver_forecast_methods(
        method_forecasts=forecasts,
        forecast_daterange=forecast_daterange,
        forecast_base=driver,
    )

    return forecasts


def forecast_drivers_methods(
    drivers: dts.DriverGroup,
    df_params: params.DriverForecastParams,
    training_daterange: datetime.date | tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    lags: dict[dts.DriverName, int] | None = None,
    max_lag: int | None = None,
    methods: list[DriverForecastingMethodEnum] | None = None,
) -> dict[dts.DriverName, dict[DriverForecastingMethodEnum, dts.Driver]]:
    if methods is None:
        methods = list(DriverForecastingMethodEnum)

    if lags is None:
        lags = {driver: 0 for driver in drivers.get_ordered_drivers()}

    if max_lag is None:
        max_lag = max(lags.values())

    forecasts: dict[dts.DriverName, dict[DriverForecastingMethodEnum, dts.Driver]] = {}

    for driver in drivers.get_ordered_drivers():
        forecasts[driver] = forecast_driver_methods(
            driver=drivers.get_driver(driver),
            df_params=df_params,
            training_daterange=training_daterange,
            forecast_daterange=forecast_daterange,
            n_lag=lags[driver],
            max_lag=max_lag,
            methods=methods,
        )

    return forecasts

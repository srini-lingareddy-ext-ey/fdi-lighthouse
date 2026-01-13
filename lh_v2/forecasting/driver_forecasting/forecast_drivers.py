import datetime
import time

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.datatypes.forecasting_types.driver_forecasting_types import (
    DriverForecastingMethodEnum,
)
from lh_v2.util import get_logger

from .driver_forecasting_methods import (
    AbstractDriverForecastingMethod,
    ARIMADriverForecastingMethod,
    AutoARIMADriverForecastingMethod,
    CrostonDriverForecastingMethod,
    CrostonTSBDriverForecastingMethod,
    ExponentialSmoothingDriverForecastingMethod,
    LinearRegressionDriverForecastingMethod,
    MovingAverageDriverForecastingMethod,
    ProphetDriverForecastingMethod,
    TBATSDriverForecastingMethod,
    VARIMADriverForecastingMethod,
    XGBoostDriverForecastingMethod,
)
from .driver_forecasting_types import DriverForecastingInput, DriverForecastingOutput

logger = get_logger(__name__)


DRIVER_FORECASTING_METHOD_MAP: dict[
    DriverForecastingMethodEnum, type[AbstractDriverForecastingMethod]
] = {
    DriverForecastingMethodEnum.LINEAR_REGRESSION: LinearRegressionDriverForecastingMethod,
    DriverForecastingMethodEnum.MOVING_AVERAGE: MovingAverageDriverForecastingMethod,
    DriverForecastingMethodEnum.ARIMA: ARIMADriverForecastingMethod,
    DriverForecastingMethodEnum.AUTO_ARIMA: AutoARIMADriverForecastingMethod,
    DriverForecastingMethodEnum.EXPONENTIAL_SMOOTHING: ExponentialSmoothingDriverForecastingMethod,
    DriverForecastingMethodEnum.XGBOOST: XGBoostDriverForecastingMethod,
    DriverForecastingMethodEnum.VARIMA: VARIMADriverForecastingMethod,
    DriverForecastingMethodEnum.PROPHET: ProphetDriverForecastingMethod,
    DriverForecastingMethodEnum.TBATS: TBATSDriverForecastingMethod,
    DriverForecastingMethodEnum.CROSTON: CrostonDriverForecastingMethod,
    DriverForecastingMethodEnum.CROSTON_TSB: CrostonTSBDriverForecastingMethod,
}


def _create_singe_driver_forecast(
    driver_info: dts.Driver,
    df_params: params.DriverForecastParams,
    n_lag: int,
    max_lag: int,
    selected_method: DriverForecastingMethodEnum,
    training_daterange: datetime.date | tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
) -> dts.Driver:
    """
    Create a forecast for a single driver using the specified forecasting method.

    Parameters
    ----------
    driver_info : dts.Driver
        The driver information containing historical data.
    df_params : params.DriverForecastParams
        Parameters for driver forecasting methods.
    n_lag : int
        Number of lag periods to use for this driver.
    max_lag : int
        Maximum lag across all drivers in the forecast.
    selected_method : DriverForecastingMethodEnum
        The forecasting method to use.
    training_daterange : datetime.date | tuple[datetime.date, datetime.date]
        Single date or date range for training data.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Date range for the forecast period.

    Returns
    -------
    dts.Driver
        Driver object containing the forecasted values.
    """
    # Initialize the forecasting method instance with driver data and parameters
    method_instance = DRIVER_FORECASTING_METHOD_MAP[selected_method](
        driver_info=driver_info,
        method_params=df_params[selected_method],
        training_date=training_daterange,
        forecast_daterange=forecast_daterange,
        n_lag=n_lag,
    )

    # Train the forecasting model and log execution time
    start_time = time.time()
    method_instance.train()
    logger.timing(
        f'Trained driver forecasting method {method_instance.name()} for driver '
        f'{method_instance.driver_info.name} in {time.time() - start_time:.4f} seconds.'
    )

    # Generate forecasts using the trained model and log execution time
    start_time = time.time()
    forecasted_driver = method_instance.get_forecasted_driver(max_lag=max_lag)
    logger.timing(
        f'Generated forecasts using method {method_instance.name()} for driver '
        f'{method_instance.driver_info.name} in {time.time() - start_time:.4f} seconds.'
    )

    return forecasted_driver


def _create_forecasts(
    drivers_info: dts.DriverGroup,
    df_params: params.DriverForecastParams,
    lags: dict[dts.DriverName, int],
    selected_method: DriverForecastingMethodEnum,
    training_daterange: datetime.date | tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
) -> dts.DriverGroup:
    """
    Create forecasts for multiple drivers using the specified forecasting method.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        Group of drivers containing historical data for forecasting.
    df_params : params.DriverForecastParams
        Parameters for driver forecasting methods.
    lags : dict[dts.DriverName, int]
        Dictionary mapping driver names to their respective lag periods.
    selected_method : DriverForecastingMethodEnum
        The forecasting method to use for all drivers.
    training_daterange : datetime.date | tuple[datetime.date, datetime.date]
        Single date or date range for training data.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Date range for the forecast period.

    Returns
    -------
    dts.DriverGroup
        DriverGroup object containing all forecasted drivers.
    """
    # Get ordered list of driver names to ensure consistent processing order
    driver_names = drivers_info.get_ordered_drivers()
    forecasted_drivers_lst: list[dts.Driver] = []

    # Determine the maximum lag across all drivers for alignment purposes
    max_lag = max(lags.values())

    # Iterate through each driver and generate individual forecasts
    for driver_name in driver_names:
        # Create forecast for current driver using specified method and parameters
        forecasted_drivers_lst.append(
            _create_singe_driver_forecast(
                driver_info=drivers_info.get_driver(driver_name),
                df_params=df_params,
                n_lag=lags[driver_name],
                max_lag=max_lag,
                selected_method=selected_method,
                training_daterange=training_daterange,
                forecast_daterange=forecast_daterange,
            )
        )

    # Combine all forecasted drivers into a single DriverGroup
    return dts.DriverGroup.from_driver_lst(forecasted_drivers_lst)


def create_driver_forecasts(
    forecasting_info: DriverForecastingInput,
    df_params: params.DriverForecastParams,
) -> DriverForecastingOutput:
    """
    Create driver forecasts using the specified forecasting method and parameters.

    This function coordinates the forecasting process by preparing the training
    date range and delegating to the appropriate forecasting method to generate
    predictions for all drivers in the input.

    Parameters
    ----------
    forecasting_info : DriverForecastingInput
        Input object containing drivers, lags, training date range, and forecast
        date range information.
    df_params : params.DriverForecastParams
        Parameters for driver forecasting, including the selected forecasting
        method and method-specific parameters.

    Returns
    -------
    DriverForecastingOutput
        Output object containing the forecasted driver values, including the
        data array, driver mapping, dates, and data type information.

    Notes
    -----
    If `forecasting_info.training_daterange` is a single date, the function
    automatically determines the minimum date from the historical driver data
    and creates a date range from that minimum date to the specified training date.
    """
    # Extract the selected forecasting method from parameters
    selected_method = df_params.selected_method

    # Determine the training date range based on input type
    if isinstance(forecasting_info.training_daterange, tuple):
        # Use the provided date range directly if already a tuple
        training_daterange = forecasting_info.training_daterange
    else:
        # If single date provided, find earliest date across all drivers
        min_date = min(
            [
                min(driver_dates.keys())
                for driver_dates in forecasting_info.drivers.dates
            ]
        )
        # Create date range from earliest available date to specified training date
        training_daterange = (
            min_date,
            forecasting_info.training_daterange,
        )

    # Generate forecasts for all drivers using the prepared parameters
    output = _create_forecasts(
        drivers_info=forecasting_info.drivers,
        df_params=df_params,
        lags=forecasting_info.lags,
        selected_method=selected_method,
        training_daterange=training_daterange,
        forecast_daterange=forecasting_info.forecast_daterange,
    )

    # Package the forecasted results into output format
    return DriverForecastingOutput(
        arr=output.arr,
        map=output.map,
        dates=output.dates,
        np_dtype=output.np_dtype,
    )

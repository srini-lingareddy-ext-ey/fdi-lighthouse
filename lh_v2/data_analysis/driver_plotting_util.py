import datetime

import matplotlib.pyplot as plt
import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.forecasting_types.driver_forecasting_types import (
    DriverForecastingMethodEnum,
)

from .plotting_util import (
    format_timeseries_for_plotting,
    format_timeseries_mapping_for_plotting,
)


def plot_driver(driver: dts.Driver):
    """
    Plot a time series of driver values over time.

    Parameters
    ----------
    driver : dts.Driver
        Driver object containing dates and array values to be plotted.

    Returns
    -------
    None
        This function displays a matplotlib plot and returns nothing.

    Notes
    -----
    Creates a 16x10 inch figure showing the driver values plotted against
    sorted dates. The plot includes a title with the driver name, labeled
    axes, and is displayed using plt.show().

    Examples
    --------
    >>> plot_driver(my_driver)
    # Displays a plot of the driver's values over time
    """
    plt.figure(figsize=(16, 10))
    plt.plot(np.array(sorted(list(driver.dates.keys()))), driver.arr)
    plt.title(f'Driver - {driver.name}')
    plt.xlabel('Date')
    plt.ylabel('Driver Value')
    plt.show()
    return


def plot_driver_forecast(
    driver: dts.Driver,
    forecast_daterange: tuple[datetime.date, datetime.date],
    driver_base: dts.Driver | None = None,
):
    """
    Plot historical and forecasted driver data.

    This function creates a visualization showing both historical driver values
    and forecasted values over a specified date range. Historical data is plotted
    as a solid line, while forecasted data is plotted as a dashed line (or as
    individual markers if only one forecast point exists).

    Parameters
    ----------
    driver : dts.Driver
        The driver object containing historical data and dates.
    forecast_daterange : tuple[datetime.date, datetime.date]
        A tuple containing the start and end dates for the forecast period.
        Format: (start_date, end_date).

    Returns
    -------
    None
        Displays a matplotlib plot showing historical and forecasted data.

    Notes
    -----
    - Historical data includes all dates before the forecast start date.
    - Forecast data includes dates within the specified forecast range (inclusive).
    - The plot uses a 16x10 inch figure size.
    - If only one forecast date exists, it's plotted as a marker instead of a line.

    Examples
    --------
    >>> import datetime
    >>> forecast_range = (datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))
    >>> plot_driver_forecast(my_driver, forecast_range)
    """
    (
        historical_dates,
        historical_values,
        forecast_dates,
        forecast_values,
        forecast_base_values,
    ) = format_timeseries_for_plotting(
        timeseries=driver,
        forecast_daterange=forecast_daterange,
        timeseries_base=driver_base,
    )

    # Create a new figure with specified dimensions
    plt.figure(figsize=(16, 10))

    # Plot historical data as a solid line
    plt.plot(
        np.array(historical_dates),
        historical_values,
        label='Historical Data',
    )

    # Plot forecast data with dashed line if multiple points, otherwise use marker
    if len(forecast_dates) > 1:
        plt.plot(
            np.array(forecast_dates),
            forecast_values,
            label='Forecasted Data',
            linestyle='--',
        )
    else:
        # For single forecast point, display as a marker without line
        plt.plot(
            np.array(forecast_dates),
            forecast_values,
            marker='o',
            label='Forecasted Data',
            linestyle='',
        )

    if forecast_base_values is not None:
        if len(forecast_dates) > 1:
            plt.plot(
                np.array(forecast_dates),
                forecast_base_values,
                label='Forecast Base',
                color='black',
                linewidth=2,
            )
        else:
            plt.plot(
                np.array(forecast_dates),
                forecast_base_values,
                marker='o',
                label='Forecast Base',
                linestyle='',
            )

    # Add title with driver name and axis labels
    plt.title(f'Driver Forecast: {driver.name}')
    plt.xlabel('Date')
    plt.ylabel('Driver Value')
    plt.legend()
    plt.show()
    return


def plot_driver_forecast_methods(
    method_forecasts: dict[DriverForecastingMethodEnum, dts.Driver],
    forecast_daterange: tuple[datetime.date, datetime.date],
    forecast_base: dts.Driver | None = None,
):
    (
        historical_dates,
        historical_values,
        forecast_dates,
        forecast_values,
        forecast_base_values,
    ) = format_timeseries_mapping_for_plotting(
        timeseries_mapping=method_forecasts,
        forecast_daterange=forecast_daterange,
        timeseries_base=forecast_base,
    )

    plt.figure(figsize=(16, 10))

    plt.plot(
        np.array(historical_dates),
        historical_values,
        label='Historical Data',
    )

    for method, forecast in forecast_values.items():
        if len(forecast_dates) > 1:
            plt.plot(
                np.array(forecast_dates),
                forecast,
                label=f'Forecasted Data - {str(method)}',
                linestyle='--',
            )
        else:
            # For single forecast point, display as a marker without line
            plt.plot(
                np.array(forecast_dates),
                forecast,
                marker='o',
                label=f'Forecasted Data - {str(method)}',
                linestyle='',
            )

    if forecast_base_values is not None:
        if len(forecast_dates) > 1:
            plt.plot(
                np.array(forecast_dates),
                forecast_base_values,
                label='Forecast Base',
                color='black',
                linewidth=2,
            )
        else:
            plt.plot(
                np.array(forecast_dates),
                forecast_base_values,
                marker='o',
                label='Forecast Base',
                linestyle='',
            )

    mappings = list(method_forecasts.keys())
    plt.title(f'Driver Forecast Comparison - {method_forecasts[mappings[0]].name}')
    plt.xlabel('Date')
    plt.ylabel('Driver Value')
    plt.legend()
    plt.show()

    return

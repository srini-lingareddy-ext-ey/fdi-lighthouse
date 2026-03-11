import datetime
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import lh_v2.datatypes as dts
from lh_v2.util import parse_snake_case

from .plotting_util import (
    format_timeseries_for_plotting,
    format_timeseries_mapping_for_plotting,
)


def plot_account(acc: dts.AccountInfo):
    """
    Plot the account value over time for a given account.

    Parameters
    ----------
    acc : dts.AccountInfo
        AccountInfo object containing account data, dates dictionary, and account type.

    Returns
    -------
    None
        Displays a matplotlib plot showing account value progression over time.

    Notes
    -----
    The function creates a 16x10 inch figure displaying the account value (arr)
    plotted against sorted dates. The account type is parsed from snake_case
    format and displayed in the title.
    """
    plt.figure(figsize=(16, 10))
    plt.plot(np.array(sorted(list(acc.dates.keys()))), acc.arr)
    plt.title(f'Account - {parse_snake_case(acc.account_type)}')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.show()
    return


def plot_account_forecast(
    acc: dts.AccountInfo,
    forecast_daterange: tuple[datetime.date, datetime.date],
    base_acc: dts.AccountInfo | None = None,
    b_plot_historicals: bool = True,
):
    """
    Plot historical and forecasted account values over time.

    This function creates a line plot showing both historical account data and
    forecasted values for a specified date range. Historical data is shown as a
    solid line, while forecasted data is shown as a dashed line (or with markers
    if only one forecast point exists).

    Parameters
    ----------
    acc : dts.AccountInfo
        An AccountInfo object containing account dates and values to be plotted.
    forecast_daterange : tuple[datetime.date, datetime.date]
        A tuple of two dates (start_date, end_date) defining the forecast period.
        The start date marks the beginning of the forecast, and the end date marks
        the end of the forecast period.

    Returns
    -------
    None
        This function displays a matplotlib plot and returns None.

    Notes
    -----
    - Historical data includes all dates before the forecast start date.
    - Forecast data includes dates within the specified forecast range (inclusive).
    - The plot size is fixed at 16x10 inches.
    - If there is only one forecasted date, it is displayed as a point with a marker
      instead of a dashed line.
    """
    # Extract all dates that occur before the forecast period begins
    (
        historical_dates,
        historical_values,
        forecast_dates,
        forecast_values,
        forecast_base,
    ) = format_timeseries_for_plotting(
        timeseries=acc, forecast_daterange=forecast_daterange, timeseries_base=base_acc
    )

    # Create a new figure with specified dimensions
    plt.figure(figsize=(16, 10))

    # Plot historical data as a solid line
    if b_plot_historicals:
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

    if forecast_base is not None:
        if len(forecast_dates) > 1:
            plt.plot(
                np.array(forecast_dates),
                forecast_base,
                label='Historical Data Over Forecast Period',
                color='black',
                linewidth=3,
                linestyle='--',
            )
        else:
            plt.plot(
                np.array(forecast_dates),
                forecast_base,
                label='Historical Data Over Forecast Period',
                color='black',
                linewidth=3,
                marker='o',
                linestyle='',
            )

    # Add title with formatted account type and axis labels
    plt.title(f'Account Forecast: {parse_snake_case(acc.account_type)}')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.show()
    return


def plot_account_forecasts(
    accounts: Mapping[Any, dts.AccountInfo],
    forecast_daterange: tuple[datetime.date, datetime.date],
    historicals: dts.AccountInfo | None = None,
    labels: Mapping[Any, str] | None = None,
    b_plot_historicals: bool = True,
):
    mappings = list(accounts.keys())

    if len(accounts) == 0:
        return
    if len(accounts) == 1:
        plot_account_forecast(
            accounts[mappings[0]], forecast_daterange, base_acc=historicals
        )
        return

    acc = accounts[mappings[0]].account_type
    for key in mappings[1:]:
        if accounts[key].account_type != acc:
            raise ValueError('All accounts must have the same account_type')

    dates = accounts[mappings[0]].dates
    for key in mappings[1:]:
        if accounts[key].dates != dates:
            raise ValueError('All accounts must have the same dates')

    (
        historical_dates,
        historical_values,
        forecast_dates,
        forecast_values,
        forecast_base_values,
    ) = format_timeseries_mapping_for_plotting(
        timeseries_mapping=accounts,
        forecast_daterange=forecast_daterange,
        timeseries_base=historicals,
    )

    # Create a new figure with specified dimensions
    plt.figure(figsize=(16, 10))

    # Plot historical data as a solid line
    if b_plot_historicals:
        plt.plot(
            np.array(historical_dates),
            historical_values,
            label='Historical Data',
        )

    # Plot forecast data with dashed line if multiple points, otherwise use marker
    if len(forecast_dates) > 1:
        for key in accounts.keys():
            if labels is not None:
                label = labels[key]
            else:
                label = str(key)
            plt.plot(
                np.array(forecast_dates),
                forecast_values[key],
                label=label,
                linestyle='--',
            )
    else:
        # For single forecast point, display as a marker without line
        for key in accounts.keys():
            if labels is not None:
                label = labels[key]
            else:
                label = str(key)
            plt.plot(
                np.array(forecast_dates),
                forecast_values[key],
                marker='o',
                label=label,
                linestyle='',
            )

    if forecast_base_values is not None:
        if len(forecast_dates) > 1:
            plt.plot(
                np.array(forecast_dates),
                forecast_base_values,
                label='Historical Data Over Forecast Period',
                color='black',
                linewidth=3,
                linestyle='--',
            )
        else:
            plt.plot(
                np.array(forecast_dates),
                forecast_base_values,
                label='Historical Data Over Forecast Period',
                color='black',
                linewidth=3,
                marker='o',
                linestyle='',
            )

    # Add title with formatted account type and axis labels
    plt.title(
        f'Account Forecast: {parse_snake_case(accounts[mappings[0]].account_type)}'
    )
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.show()
    return

import datetime

import matplotlib.pyplot as plt
import numpy as np

import lh_v2.datatypes as dts


def parse_snake_case(s: str) -> str:
    """
    Convert a snake_case string to Title Case with spaces.

    Parameters
    ----------
    s : str
        A string in snake_case format (words separated by underscores).

    Returns
    -------
    str
        A string in Title Case format with spaces replacing underscores.

    Examples
    --------
    >>> parse_snake_case('hello_world')
    'Hello World'
    >>> parse_snake_case('my_variable_name')
    'My Variable Name'
    """
    return s.replace('_', ' ').title()


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
    acc: dts.AccountInfo, forecast_daterange: tuple[datetime.date, datetime.date]
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
    historical_dates = [
        date for date in acc.dates.keys() if date < forecast_daterange[0]
    ]
    # Get the account values for the historical period
    historical_values = acc.apply_daterange(end_date=historical_dates[-1]).arr

    # Extract all dates within the forecast date range (inclusive)
    forecast_dates = [
        date
        for date in acc.dates.keys()
        if forecast_daterange[0] <= date <= forecast_daterange[1]
    ]
    # Get the account values for the forecast period
    forecast_values = acc.apply_daterange(
        start_date=forecast_daterange[0], end_date=forecast_daterange[1]
    ).arr

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

    # Add title with formatted account type and axis labels
    plt.title(f'Account Forecast: {parse_snake_case(acc.account_type)}')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.show()
    return


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
    driver: dts.Driver, forecast_daterange: tuple[datetime.date, datetime.date]
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
    # Extract all dates that occur before the forecast period begins
    historical_dates = [
        date for date in driver.dates.keys() if date < forecast_daterange[0]
    ]
    # Get the driver values for the historical period
    historical_values = driver.apply_daterange(end_date=historical_dates[-1]).arr

    # Extract all dates within the forecast date range (inclusive)
    forecast_dates = [
        date
        for date in driver.dates.keys()
        if forecast_daterange[0] <= date <= forecast_daterange[1]
    ]
    # Get the driver values for the forecast period
    forecast_values = driver.apply_daterange(
        start_date=forecast_daterange[0], end_date=forecast_daterange[1]
    ).arr

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

    # Add title with driver name and axis labels
    plt.title(f'Driver Forecast: {driver.name}')
    plt.xlabel('Date')
    plt.ylabel('Driver Value')
    plt.legend()
    plt.show()
    return

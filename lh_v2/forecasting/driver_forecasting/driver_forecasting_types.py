import datetime
from dataclasses import dataclass

from dateutil.relativedelta import relativedelta

import lh_v2.datatypes as dts
from lh_v2.io.plotting import plot_driver_forecast
from lh_v2.util import month_dif


@dataclass
class DriverForecastingInput:
    """
    Input data structure for driver forecasting.

    Attributes
    ----------
    drivers : dts.DriverGroup
        The group of drivers for each account, containing information about
        which drivers are associated with which accounts.
    lags : dict[dts.DriverName, int]
        A dictionary mapping driver names to the number of lag periods to apply.
        Keys are driver names, and values are integers representing the number
        of periods to lag.
    training_daterange : datetime.date | tuple[datetime.date, datetime.date]
        The date range for training data. Can be either a single date or a tuple
        of (start_date, end_date).
    forecast_daterange : tuple[datetime.date, datetime.date]
        The date range for forecasting as a tuple of (start_date, end_date).
    """

    drivers: dts.DriverGroup
    lags: dict[dts.DriverName, int]
    training_daterange: datetime.date | tuple[datetime.date, datetime.date]
    forecast_daterange: tuple[datetime.date, datetime.date]


@dataclass
class DriverForecastingOutput(dts.DriverGroup):
    """
    Output container for driver forecasting results.

    This class extends DriverGroup to provide a structured output format
    for driver forecasting operations. It inherits all functionality from the
    parent class while serving as a distinct type for forecast outputs.

    Attributes
    ----------
    Inherits all attributes from dts.DriverGroup

    See Also
    --------
    dts.DriverGroup : Parent class providing core driver group functionality

    Notes
    -----
    This class currently serves as a pass-through implementation of DriverGroup,
    specifically typed for driver forecasting output purposes.

    Examples
    --------
    >>> output = DriverForecastingOutput()
    >>> # Use as an DriverGroup for forecast results
    """

    def plot_driver_forecasts(
        self,
        driver_name: dts.DriverName,
        lag: int,
        forecast_daterange: tuple[datetime.date, datetime.date],
    ) -> None:
        """
        Plot historical and forecasted values for a specified driver.

        This method visualizes both the historical data and forecasted values for
        a driver, with the forecast period highlighted. It automatically adjusts
        the forecast date range based on the driver's lag value.

        Parameters
        ----------
        driver_name : dts.DriverName
            The name of the driver to plot.
        lag : int
            The lag value (in months) applied to this driver. Used to adjust the
            forecast date range to account for temporal offset.
        forecast_daterange : tuple[datetime.date, datetime.date]
            The date range for the forecast as (start_date, end_date). Will be
            adjusted by subtracting the lag from the end date.

        Returns
        -------
        None
            Displays a plot of historical and forecasted driver values. Returns
            early without plotting if lag exceeds the forecast date range length.

        Notes
        -----
        If the lag is greater than the number of months in the forecast date range,
        the function prints a message and returns without plotting, as no forecasting
        is required in this case.

        The forecast date range is adjusted by subtracting the lag (in months) from
        the end date to account for the temporal offset of lagged drivers.

        See Also
        --------
        lh_v2.io.plotting.plot_driver_forecast : Underlying plotting function.

        Examples
        --------
        >>> output = DriverForecastingOutput(...)
        >>> output.plot_driver_forecasts(
        ...     driver_name=DriverName('temperature'),
        ...     lag=3,
        ...     forecast_daterange=(date(2024, 1, 1), date(2024, 12, 31))
        ... )
        """

        if lag > month_dif(forecast_daterange[0], forecast_daterange[1]):
            print(
                'Lag is greater than the forecast date range. No forecasting requiered.'
            )
            return

        forecast_daterange = (
            forecast_daterange[0],
            forecast_daterange[1] - relativedelta(months=lag),
        )

        driver = self.get_driver(driver_name)

        plot_driver_forecast(
            driver=driver,
            forecast_daterange=forecast_daterange,
        )
        return

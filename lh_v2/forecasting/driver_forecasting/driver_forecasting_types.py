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
    account_drivers : dts.AccountDriverGroup
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
        Plots the historical and forecasted values for a specified driver.

        Parameters
        ----------
        driver_name : dts.DriverName
            The name of the driver to plot.
        forecast_daterange : tuple[datetime.date, datetime.date]
            The date range for the forecast to be highlighted on the plot.

        Returns
        -------
        None
            Displays a plot of historical and forecasted driver values.
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

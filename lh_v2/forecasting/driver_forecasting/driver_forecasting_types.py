import datetime
from dataclasses import dataclass

import lh_v2.datatypes as dts


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

    pass

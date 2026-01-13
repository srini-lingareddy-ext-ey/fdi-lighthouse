import datetime
from abc import ABC, abstractmethod
from typing import Any, Optional

from dateutil.relativedelta import relativedelta

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif


class AbstractDriverForecastingMethod(ABC):
    """
    Abstract base class for driver forecasting methods.

    This class defines the interface and common functionality for all driver
    forecasting implementations. Subclasses must implement the `name`, `train`,
    and `apply` methods to provide specific forecasting algorithms.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver object containing historical data and metadata.
    method_params : params.forecasting_params.BaseDriverForecastParams
        Configuration parameters for the forecasting method.
    training_date : datetime.date or tuple of datetime.date
        The date or date range used for training the model. If a tuple, represents
        (start_date, end_date); if a single date, represents the end_date only.
    forecast_daterange : tuple of datetime.date
        The (start_date, end_date) range for which forecasts are required.
    n_lag : int
        Number of lag periods to apply when generating forecasts.

    Attributes
    ----------
    driver_info : dts.Driver
        Driver object containing historical data and metadata.
    method_params : params.forecasting_params.BaseDriverForecastParams
        Configuration parameters for the forecasting method.
    training_date : datetime.date or tuple of datetime.date
        The date or date range used for training the model.
    forecast_daterange : tuple of datetime.date
        The (start_date, end_date) range for which forecasts are required.
    n_lag : int
        Number of lag periods to apply when generating forecasts.
    model : Any or None
        The trained forecasting model. None until `train()` is called.

    Methods
    -------
    name()
        Return the name identifier for the forecasting method.
    get_last_historical_date()
        Get the date of the last available historical data point.
    n_forecast_values_required()
        Calculate the number of forecast values needed.
    need_forecast()
        Determine if forecasting is required based on available data.
    forecast_dates_required()
        Generate list of dates requiring forecasts.
    get_training_data()
        Retrieve training data for the specified date range.
    train()
        Train the forecasting model (must be implemented by subclasses).
    apply()
        Apply the trained model to generate forecasts (must be implemented by subclasses).
    pull_full_from_historicals(max_lag)
        Pull complete historical data with appropriate date range.
    get_forecasted_driver(max_lag)
        Get the complete driver with historical and forecasted values.

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly. Subclasses
    must implement the abstract methods `name`, `train`, and `apply` to provide
    concrete forecasting implementations.
    """

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.BaseDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        self.driver_info = driver_info
        self.method_params = method_params
        self.training_date = training_date
        self.forecast_daterange = forecast_daterange
        self.n_lag = n_lag
        self.model: Optional[Any] = None
        return

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Return the name of the forecasting method.

        Returns
        -------
        str
            The name identifier for this forecasting method implementation.
        """
        raise NotImplementedError()

    def get_last_historical_date(self) -> datetime.date:
        """
        Get the last date of historical data before the forecast period begins.

        This method calculates the last historical date by subtracting one month
        from the first date in the forecast date range.

        Returns
        -------
        datetime.date
            The last date of the historical period, which is one month before
            the start of the forecast date range.

        Examples
        --------
        >>> forecaster.forecast_daterange = [datetime.date(2024, 2, 1), ...]
        >>> forecaster.get_last_historical_date()
        datetime.date(2024, 1, 1)
        """
        return self.forecast_daterange[0] - relativedelta(months=1)

    def n_forecast_values_required(self) -> int:
        """
        Calculate the number of forecast values required.

        This method computes the number of forecast periods needed by calculating
        the difference in months between the last historical date and the end of
        the forecast date range, then subtracting the lag period.

        Returns
        -------
        int
            The number of forecast values required, adjusted for the lag period.

        Notes
        -----
        The calculation accounts for:
        - The time span from the last historical date to the forecast end date
        - The lag period (n_lag) which represents the number of periods to offset
        """
        return (
            month_dif(
                self.get_last_historical_date(),
                self.forecast_daterange[1],
            )
            - self.n_lag
        )

    def need_forecast(self) -> bool:
        """
        Determine if forecasting is needed based on available data and target date range.

        This method checks whether the available historical data is sufficient to cover
        the forecast period, or if additional forecasting is required. It compares the
        lag period (n_lag) with the time difference between the most recent available
        data point and the end of the forecast period.

        Returns
        -------
        bool
            True if forecasting is needed (i.e., the lag period is less than the
            time difference between the latest available date and the end of the
            forecast period), False otherwise.

        Notes
        -----
        The method uses the `month_dif` function to calculate the difference in months
        between the maximum date in `driver_info.dates` and the end date of the
        forecast range (`forecast_daterange[1]`).
        """
        return self.n_forecast_values_required() > 0

    def forecast_dates_required(self) -> list[datetime.date]:
        """
        Calculate the list of dates that require forecasting.

        This method generates a list of future dates for which forecasts are needed,
        starting from the month after the last available historical date and extending
        for the number of forecast values required.

        Returns
        -------
        list[datetime.date]
            A list of dates for which forecasts are required. Returns an empty list
            if no forecast is needed (when `need_forecast()` returns False).

        Notes
        -----
        The dates are generated on a monthly basis using `relativedelta`, with each
        date being one month after the previous one.

        Examples
        --------
        If the last historical date is 2024-01-31 and 3 forecast values are required,
        this method would return dates for February, March, and April 2024.
        """
        if not self.need_forecast():
            return []

        last_available_date = self.get_last_historical_date()
        return [
            last_available_date + relativedelta(months=i)
            for i in range(1, self.n_forecast_values_required() + 1)
        ]

    def get_training_data(self) -> ArrayF:
        """
        Retrieve training data for the specified date range.

        This method extracts training data from the driver_info object based on
        the training_date attribute. If training_date is a tuple, it uses both
        start and end dates; otherwise, it uses only the end date.

        Returns
        -------
        ArrayF
            A numpy array containing the training data for the specified date range.

        Notes
        -----
        The method calls `apply_daterange` on the `driver_info` object with either:
        - Both start_date and end_date if `training_date` is a tuple
        - Only end_date if `training_date` is a single value
        """
        # Check if training_date is a tuple containing start and end dates
        if isinstance(self.training_date, tuple):
            # Extract data using both start and end dates from the tuple
            training_data = self.driver_info.apply_daterange(
                start_date=self.training_date[0],
                end_date=self.training_date[1],
            )
        else:
            # Extract data using only the end date (start date defaults to earliest available)
            training_data = self.driver_info.apply_daterange(
                end_date=self.training_date,
            )
        # Return the underlying numpy array from the Driver object
        return training_data.arr

    @abstractmethod
    def train(self) -> None:
        """
        Train the forecasting model.

        This method should be implemented by subclasses to define the training
        procedure for the specific forecasting model.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        Subclasses should override this method to implement their specific
        training logic, which may include fitting models to historical data,
        tuning hyperparameters, and validating model performance.
        """
        raise NotImplementedError()

    def _final_output_start_date(self, max_lag: int) -> datetime.date:
        """
        Calculate the start date for the final output based on maximum lag.

        This method determines the starting date for forecast output by adjusting
        for the maximum lag and the number of lag periods used in the model.

        Parameters
        ----------
        max_lag : int
            The maximum lag period to consider when calculating the start date.

        Returns
        -------
        datetime.date
            The calculated start date for the final output. If training_date is a tuple,
            uses the first element as the base date. Otherwise, uses the minimum date
            from driver_info.dates as the base date. The returned date is adjusted by
            adding (max_lag - n_lag) months to the base date.

        Notes
        -----
        The calculation uses relativedelta to add months, which properly handles
        month-end dates and varying month lengths.
        """
        if isinstance(self.training_date, tuple):
            return self.training_date[0] + relativedelta(months=max_lag - self.n_lag)
        else:
            return min(self.driver_info.dates.keys()) + relativedelta(
                months=max_lag - self.n_lag
            )

    def pull_full_from_historicals(self, max_lag: int) -> dts.Driver:
        """
        Pull historical driver data with appropriate lag considerations.

        This method retrieves the full historical driver data by applying a date range
        that accounts for the maximum lag required and the forecast end date adjusted
        by the number of lag months.

        Parameters
        ----------
        max_lag : int
            The maximum lag period in months to consider when determining the start
            date for historical data retrieval.

        Returns
        -------
        dts.Driver
            A Driver object containing the historical data within the calculated
            date range, from the final output start date (adjusted for max_lag) to
            the forecast end date minus n_lag months.

        Notes
        -----
        The start date is determined by `_final_output_start_date(max_lag)` and the
        end date is calculated as the forecast end date minus `n_lag` months.
        """
        return self.driver_info.apply_daterange(
            start_date=self._final_output_start_date(max_lag=max_lag),
            end_date=self.forecast_daterange[1] - relativedelta(months=self.n_lag),
        )

    @abstractmethod
    def apply(
        self,
    ) -> ArrayF:
        """
        Applies the trained model to generate forecasts.

        This method should return an ArrayF containing only the forecasted driver values.
        If there are no requiered forecasted values, it should return an empty ArrayF.

        Returns
        -------
        ArrayF
            shape (self.n_forecast_values_required(),)

        """
        raise NotImplementedError()

    def get_forecasted_driver(self, max_lag: int) -> dts.Driver:
        """
        Get the forecasted driver values, either from historical data or model predictions.

        This method returns a Driver object containing either historical values (if no
        forecast is needed) or a combination of historical and forecasted values (if
        forecasting is required).

        Parameters
        ----------
        max_lag : int
            The maximum number of periods any driver in the batch of drivers has.
            Used to determine the start date for the final output.

        Returns
        -------
        dts.Driver
            A Driver object containing the historical and/or forecasted values for the
            specified date range.

        Raises
        ------
        ValueError
            If the model has not been trained before attempting to generate forecasts.
        AssertionError
            If historical data is missing for the date prior to the forecast start date,
            which would result in missing dates in the time series.

        Notes
        -----
        The method follows this logic:
        1. If no forecast is needed, returns historical data only
        2. Validates that the model has been trained
        3. Verifies that historical data exists for the date prior to forecast start
        4. Generates forecasted values using the trained model
        5. Combines historical and forecasted data into a single Driver object
        """
        # If we already have enough historical data to cover the forecast period,
        # just return the relevant historical values without forecasting
        if not self.need_forecast():
            return self.pull_full_from_historicals(max_lag=max_lag)

        # Ensure the model has been trained before attempting to generate forecasts
        if self.model is None:
            raise ValueError(
                f'Model has not been trained for method {self.name()}. '
                'Call train() before applying the model.'
            )

        # Verify that we have historical data for the month immediately before
        # the forecast start date to ensure continuity in the time series
        assert (
            self.forecast_daterange[0] - relativedelta(months=1)
        ) in self.driver_info.dates.keys(), (
            'Historical data missing for date prior to forecast start date. '
            'Cannot create time series with missing dates.'
        )

        # Generate the list of dates that need to be forecasted
        forecasted_dates = self.forecast_dates_required()

        # Apply the trained model to generate forecasted values
        forecasted_values = self.apply()

        # Combine historical data (from appropriate start date) with new forecasts
        # and return as a complete Driver object spanning the full date range
        return self.driver_info.apply_daterange(
            start_date=self._final_output_start_date(max_lag=max_lag)
        ).add_forecast_vals(
            forecast_dates=forecasted_dates, forecast_values=forecasted_values
        )

import datetime

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.params.forecasting_params import MovingAverageDriverForecastParams
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif

from .abstract_class import AbstractDriverForecastingMethod


class MovingAverageDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    Moving average-based driver forecasting method.

    Uses a rolling window of historical values to calculate a simple moving average,
    which is then used as a constant forecast for all future periods.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.MovingAverageDriverForecastParams
        Parameters specific to the moving average forecasting method.
    training_date : datetime.date | tuple[datetime.date, datetime.date]
        Single date or date range (start, end) for training the model.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Date range (start, end) for which forecasts should be generated.
    n_lag : int
        Number of months to lag the forecast.
    """

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.MovingAverageDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        # Validate params are correct type
        assert isinstance(method_params, MovingAverageDriverForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{MovingAverageDriverForecastParams}, Params were of type {type(method_params)}'
        )

        # Store forecasting configuration
        self.driver_info = driver_info
        self.method_params = method_params
        self.training_date = training_date
        self.forecast_daterange = forecast_daterange
        self.n_lag = n_lag

        # Moving average will be calculated during training
        self.model: float | None = None
        self.moving_average: float | None = None

    @staticmethod
    def name() -> str:
        return 'Moving Average Driver Forecasting Method'

    def train(self) -> None:
        """
        Calculate the moving average from the training data.

        Computes a simple moving average over the last 'window_size' periods,
        which is used as a constant forecast for all future periods.
        """
        # Determine training date range
        if isinstance(self.training_date, tuple):
            training_start, training_end = self.training_date
        else:
            training_end = self.training_date
            training_start = min(self.driver_info.dates.keys())

        n_forecast_needed = month_dif(
            self.forecast_daterange[0], self.forecast_daterange[1]
        )

        if self.n_lag >= n_forecast_needed:
            return

        training_driver = self.driver_info.apply_daterange(
            start_date=training_start, end_date=training_end
        )

        # Calculate moving average from last 'window_size' periods
        window_size = self.method_params.window_size
        if len(training_driver.arr) < window_size:
            avg_value = float(np.mean(training_driver.arr))
        else:
            avg_value = float(np.mean(training_driver.arr[-window_size:]))

        self.model = avg_value
        self.moving_average = avg_value

    def apply(self) -> ArrayF:
        """
        Apply the moving average to generate forecasts.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.
            Moving average produces constant (flat) forecasts.

        Raises
        ------
        ValueError
            If the model has not been trained when forecasting is required.
        """
        if not self.need_forecast():
            return np.ndarray((0,), dtype=self.driver_info.np_dtype)

        if self.model is None:
            raise ValueError(f'Model not trained for method {self.name()}.')

        # Generate flat forecast using the calculated moving average
        n_forecast = self.n_forecast_values_required()
        return np.full(n_forecast, self.model, dtype=np.float64)

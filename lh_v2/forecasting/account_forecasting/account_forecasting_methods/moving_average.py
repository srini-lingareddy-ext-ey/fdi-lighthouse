import datetime
from typing import Optional

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import MovingAverageAccountForecastParams
from lh_v2.shared import ArrayF

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class MovingAverageAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    Moving Average forecasting method for account data.

    Uses a rolling window of historical account values to calculate a simple moving average,
    which is then used as a constant forecast for all future periods. This is a time-series
    method that does not require driver data.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        Account and driver information (only account data is used).
    best_lags : dict[dts.DriverName, int]
        Optimal lag for each driver (not used by this method).
    training_daterange : tuple[datetime.date, datetime.date]
        Training period start and end dates.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Forecast period start and end dates.
    model_params : MovingAverageAccountForecastParams
        Parameters specific to Moving Average forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: MovingAverageAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, MovingAverageAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{MovingAverageAccountForecastParams}, Params were of type {type(model_params)}'
        )

        # Store configuration
        self.info = info
        self.best_lags = best_lags
        self.training_daterange = training_daterange
        self.forecast_daterange = forecast_daterange
        self.model_params = model_params
        self.training_info: Optional[dts.AccountDriverGroup] = None
        self.forecasting_input_info: Optional[dts.AccountDriverGroup] = None
        self.validation_info: Optional[dts.AccountInfo] = None

        # Moving average will be calculated during training
        self.model: float | None = None

    @staticmethod
    def name() -> str:
        return 'Moving Average Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.MOVING_AVERAGE

    @staticmethod
    def is_linear() -> bool:
        return False

    def train(self) -> None:
        """
        Calculate the moving average from the training data.

        Computes a simple moving average over the last 'window_size' periods
        from the account's historical data, which is used as a constant forecast
        for all future periods.
        """
        training_info = self.get_training_data()

        # Get account training data
        account_arr = training_info.account.arr

        # Calculate moving average from last 'window_size' periods
        window_size = self.model_params.window_size

        if len(account_arr) < window_size:
            # If we don't have enough data, use mean of all available data
            avg_value = float(np.mean(account_arr))
        else:
            # Use mean of last window_size periods
            avg_value = float(np.mean(account_arr[-window_size:]))

        self.model = avg_value

    def apply(self) -> ArrayF:
        """
        Apply the moving average to generate forecasts.

        Returns
        -------
        ArrayF
            Predicted account values for the forecast period.
            Moving average produces constant (flat) forecasts.

        Raises
        ------
        ModelNotTrainedError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ModelNotTrainedError(method_name=self.name())

        # Get number of forecast periods from forecast dates
        n_forecast = len(self.get_forecasting_dates())

        # Generate flat forecast using the calculated moving average
        return np.full(n_forecast, self.model, dtype=np.float64)

    def apply_vectorized(self, arr_input: ArrayF) -> ArrayF:
        """
        Apply the moving average to a vectorized input array.

        The input array should be of shape (n_forecasts, n_features, n_samples),
        but since Moving Average does not use features, it will ignore the
        n_features dimension and produce a flat forecast for each sample.

        Returns
        -------
        ArrayF
            Predicted account values for the forecast period, shape (n_forecasts, n_samples).
            Each forecast is a constant value equal to the moving average.

        Raises
        ------
        ModelNotTrainedError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ModelNotTrainedError(method_name=self.name())

        n_forecasts, _, n_samples = arr_input.shape

        # Generate flat forecast using the calculated moving average
        return np.full((n_forecasts, n_samples), self.model, dtype=np.float64)

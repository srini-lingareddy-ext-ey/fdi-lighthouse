import datetime
from typing import Any, Optional

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import ExponentialSmoothingAccountForecastParams
from lh_v2.shared import ArrayF

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class ExponentialSmoothingAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    Exponential Smoothing (Holt-Winters) forecasting method for account data.

    Simple and robust method that handles trend and seasonality. Automatically detects
    trend and seasonality components when enabled. This is a time-series method that
    does not require driver data.

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
    model_params : ExponentialSmoothingAccountForecastParams
        Parameters specific to Exponential Smoothing forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: ExponentialSmoothingAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, ExponentialSmoothingAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{ExponentialSmoothingAccountForecastParams}, Params were of type {type(model_params)}'
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

        # Model will be trained later
        self.model: Any | None = None
        self.fitted_model: Any | None = None
        self.seasonal_strength = 0.0

    @staticmethod
    def name() -> str:
        return 'Exponential Smoothing Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.EXPONENTIAL_SMOOTHING

    def _detect_seasonality(self, data: ArrayF) -> tuple[bool, float]:
        """
        Detect if time series has significant seasonality.

        Parameters
        ----------
        data : ArrayF
            Historical time series data.

        Returns
        -------
        tuple[bool, float]
            (has_seasonality, seasonal_strength)
        """
        if len(data) < 2 * self.model_params.seasonal_periods:
            return False, 0.0

        try:
            # Decompose into trend, seasonal, residual
            decomp = seasonal_decompose(
                data,
                model='additive',
                period=self.model_params.seasonal_periods,
                extrapolate_trend='freq',  # type: ignore[arg-type]
            )

            # Calculate seasonal strength (ratio of seasonal variance to total variance)
            seasonal_strength = float(np.var(decomp.seasonal) / np.var(data))

            return bool(
                seasonal_strength > self.model_params.seasonality_threshold
            ), seasonal_strength
        except Exception:
            return False, 0.0

    def train(self) -> None:
        """
        Fit Exponential Smoothing model to training data with auto-detection.

        Automatically detects and applies trend and seasonality components when enabled.
        Falls back to simple exponential smoothing if fitting fails.
        """
        training_info = self.get_training_data()

        # Get account training data
        account_arr = training_info.account.arr

        # Make local copies of parameters that might be auto-detected
        trend = self.model_params.trend
        seasonal = self.model_params.seasonal

        # Detect trend and seasonality components if enabled
        if self.model_params.auto_detect:
            has_seasonality, self.seasonal_strength = self._detect_seasonality(
                account_arr
            )

            if trend is None:
                trend = 'add'

            if seasonal is None:
                seasonal = 'add' if has_seasonality else None

        # Fit Exponential Smoothing model with detected/configured components
        try:
            if (
                seasonal is not None
                and len(account_arr) >= 2 * self.model_params.seasonal_periods
            ):
                model = ExponentialSmoothing(
                    account_arr,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=self.model_params.seasonal_periods,
                )
            else:
                model = ExponentialSmoothing(
                    account_arr,
                    trend=trend,
                    seasonal=None,
                )

            self.fitted_model = model.fit()
            self.model = self.fitted_model

        except Exception:
            # Fall back to simple exponential smoothing if fitting fails
            model = ExponentialSmoothing(
                account_arr,
                trend=None,
                seasonal=None,
            )
            self.fitted_model = model.fit()
            self.model = self.fitted_model

    def apply(self) -> ArrayF:
        """
        Generate forecasts using the fitted Exponential Smoothing model.

        Returns
        -------
        ArrayF
            Predicted account values for the forecast period.

        Raises
        ------
        ModelNotTrainedError
            If the model has not been trained yet.
        """
        if self.fitted_model is None:
            raise ModelNotTrainedError(method_name=self.name())

        # Get number of forecast periods from forecast dates
        n_forecast = len(self.get_forecasting_dates())

        # Generate forecasts using fitted Exponential Smoothing model
        forecast = self.fitted_model.forecast(steps=n_forecast)
        return np.array(forecast)

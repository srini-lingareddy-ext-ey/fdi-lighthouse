import datetime
from typing import Any

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractDriverForecastingMethod


class ExponentialSmoothingDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    Exponential Smoothing (Holt-Winters) driver forecasting method.

    Simple and robust method that handles trend and seasonality. Automatically detects
    trend and seasonality components when enabled.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.ExponentialSmoothingDriverForecastParams
        Parameters specific to the Exponential Smoothing method.
    training_date : datetime.date | tuple[datetime.date, datetime.date]
        Training date or date range.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Forecast period start and end dates.
    n_lag : int
        Number of lag periods.
    """

    SEASONAL_KEYWORDS = ['Degree Days', 'Heating', 'Cooling', 'Temperature', 'Weather']

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.ExponentialSmoothingDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        super().__init__(
            driver_info, method_params, training_date, forecast_daterange, n_lag
        )

        # Extract Exponential Smoothing parameters from method_params
        self.trend = method_params.trend
        self.seasonal = method_params.seasonal
        self.seasonal_periods = method_params.seasonal_periods
        self.auto_detect = method_params.auto_detect
        self.seasonality_threshold = method_params.seasonality_threshold

        # Model will be trained later
        self.fitted_model: Any | None = None
        self.seasonal_strength = 0.0

    @staticmethod
    def name() -> str:
        return 'Exponential Smoothing'

    def _detect_seasonality(self, data: ArrayF) -> tuple[bool, float]:
        """
        Detect if time series has significant seasonality.

        Returns
        -------
        tuple[bool, float]
            (has_seasonality, seasonal_strength)
        """
        if len(data) < 2 * self.seasonal_periods:
            return False, 0.0

        try:
            # Decompose into trend, seasonal, residual
            decomp = seasonal_decompose(
                data,
                model='additive',
                period=self.seasonal_periods,
                extrapolate_trend='freq',  # type: ignore[arg-type]
            )

            # Calculate seasonal strength (ratio of seasonal variance to total variance)
            seasonal_strength = float(np.var(decomp.seasonal) / np.var(data))

            return bool(
                seasonal_strength > self.seasonality_threshold
            ), seasonal_strength
        except Exception:
            return False, 0.0

    def _classify_driver(self, data: ArrayF) -> tuple[bool, float]:
        """
        Determine if driver should use seasonal exponential smoothing.

        Returns
        -------
        tuple[bool, float]
            (use_seasonal, seasonal_strength)
        """
        # Check for seasonal keywords in driver name
        has_seasonal_keyword = any(
            keyword in self.driver_info.name for keyword in self.SEASONAL_KEYWORDS
        )

        # Detect seasonal pattern statistically
        has_seasonal_pattern, strength = self._detect_seasonality(data)

        # Use seasonality if either condition is met
        use_seasonal = has_seasonal_keyword or has_seasonal_pattern

        return use_seasonal, strength

    def train(self) -> None:
        """
        Fit Exponential Smoothing model to training data with auto-detection.

        Automatically detects and applies trend and seasonality components when enabled.
        Falls back to simple exponential smoothing if fitting fails.
        """
        if not self.need_forecast():
            return

        training_data = self.get_training_data()

        # Detect trend and seasonality components if enabled
        if self.auto_detect:
            use_seasonal, self.seasonal_strength = self._classify_driver(training_data)

            if self.trend is None:
                self.trend = 'add'

            if self.seasonal is None:
                if use_seasonal:
                    self.seasonal = 'add'
                    print(
                        f"Exponential Smoothing: Detected seasonality in '{self.driver_info.name}' "
                        f'(strength: {self.seasonal_strength:.2%})'
                    )
                else:
                    self.seasonal = None
                    print(
                        f"Exponential Smoothing: No significant seasonality in '{self.driver_info.name}' "
                        f'(strength: {self.seasonal_strength:.2%})'
                    )

        # Fit Exponential Smoothing model with detected/configured components
        try:
            if (
                self.seasonal is not None
                and len(training_data) >= 2 * self.seasonal_periods
            ):
                model = ExponentialSmoothing(
                    training_data,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                )
            else:
                model = ExponentialSmoothing(
                    training_data,
                    trend=self.trend,
                    seasonal=None,
                )

            self.fitted_model = model.fit()
            self.model = self.fitted_model

        except Exception as e:
            # Fall back to simple exponential smoothing if fitting fails
            print(
                f'Exponential Smoothing fitting failed with trend={self.trend}, seasonal={self.seasonal}: {e}'
            )
            print(
                'Falling back to simple exponential smoothing (no trend, no seasonality)'
            )

            model = ExponentialSmoothing(
                training_data,
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
            Forecasted driver values for the required forecast dates.

        Raises
        ------
        RuntimeError
            If the model has not been trained when forecasting is required.
        """
        n_forecast = self.n_forecast_values_required()

        if n_forecast <= 0:
            return np.array([])

        if self.fitted_model is None:
            raise RuntimeError('Model must be trained before forecasting')

        # Generate forecasts using fitted Exponential Smoothing model
        forecast = self.fitted_model.forecast(steps=n_forecast)
        return np.array(forecast)

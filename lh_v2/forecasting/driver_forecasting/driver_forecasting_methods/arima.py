import datetime
from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractDriverForecastingMethod


class ARIMADriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    ARIMA-based driver forecasting method with automatic seasonality detection.

    Uses ARIMA/SARIMA to capture autocorrelation and trends. Automatically detects
    and incorporates seasonality when statistically significant.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.ARIMADriverForecastParams | None
        Parameters specific to the ARIMA forecasting method.
    training_date : datetime.date | tuple[datetime.date, datetime.date]
        Training date or date range.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Forecast period start and end dates.
    n_lag : int
        Number of lag periods.
    """

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.ARIMADriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        super().__init__(
            driver_info, method_params, training_date, forecast_daterange, n_lag
        )

        # Extract ARIMA parameters from method_params
        self.order = method_params.order
        self.auto_seasonality = method_params.auto_seasonality
        self.seasonality_threshold = method_params.seasonality_threshold
        self.seasonal_period = method_params.seasonal_period

        # Model will be trained later
        self.fitted_model: Any | None = None
        self.seasonal_strength = 0.0

        # Determine seasonality configuration
        if self.auto_seasonality and method_params.seasonal_order is None:
            self.seasonal_order = None
        else:
            self.seasonal_order = method_params.seasonal_order

    @staticmethod
    def name() -> str:
        return 'ARIMA'

    def _detect_seasonality(self, data: ArrayF) -> tuple[bool, float]:
        """
        Detect if time series has significant seasonality.

        Returns
        -------
        tuple[bool, float]
            (has_seasonality, seasonal_strength)
        """
        if len(data) < 2 * self.seasonal_period:
            return False, 0.0

        try:
            # Decompose into trend, seasonal, residual
            # Note: statsmodels accepts 'freq' but type stubs say int only
            decomp = seasonal_decompose(
                data,
                model='additive',
                period=self.seasonal_period,
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
        Determine if driver should use seasonal ARIMA.

        Returns
        -------
        tuple[bool, float]
            (use_seasonal, seasonal_strength)
        """
        # Detect seasonal pattern statistically
        use_seasonal, strength = self._detect_seasonality(data)

        return use_seasonal, strength

    def train(self) -> None:
        """
        Fit ARIMA model to training data with automatic seasonality detection.

        Automatically detects and applies SARIMA when seasonality is statistically
        significant. Falls back to simpler model parameters if fitting fails.
        """
        if not self.need_forecast():
            return

        training_data = self.get_training_data()

        # Detect seasonality statistically and configure SARIMA if needed
        if self.auto_seasonality and self.seasonal_order is None:
            use_seasonal, self.seasonal_strength = self._classify_driver(training_data)

            if use_seasonal:
                self.seasonal_order = (1, 0, 0, self.seasonal_period)
                print(
                    f"ARIMA: Detected seasonality in '{self.driver_info.name}' "
                    f'(strength: {self.seasonal_strength:.2%}), using SARIMA'
                )
            else:
                print(
                    f"ARIMA: No significant seasonality in '{self.driver_info.name}' "
                    f'(strength: {self.seasonal_strength:.2%}), using non-seasonal ARIMA'
                )

        # Fit ARIMA/SARIMA model with configured parameters
        try:
            arima_model = ARIMA(
                training_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = arima_model.fit()
            self.model = self.fitted_model
        except Exception as e:
            # Fall back to simpler ARIMA(1,1,0) if fitting fails
            print(f'ARIMA fitting failed with order {self.order}: {e}')
            print('Falling back to (1, 1, 0) without seasonality')
            self.order = (1, 1, 0)
            arima_model = ARIMA(
                training_data,
                order=self.order,
                seasonal_order=None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = arima_model.fit()
            self.model = self.fitted_model

    def apply(self) -> ArrayF:
        """
        Generate forecasts using the fitted ARIMA model.

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

        # Generate forecasts using fitted ARIMA/SARIMA model
        forecast = self.fitted_model.forecast(steps=n_forecast)
        return np.array(forecast)

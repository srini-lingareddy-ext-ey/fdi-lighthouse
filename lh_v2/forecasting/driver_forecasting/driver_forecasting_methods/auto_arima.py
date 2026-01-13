import datetime
from typing import Any

import numpy as np
from pmdarima import auto_arima

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractDriverForecastingMethod


class AutoARIMADriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    Auto-ARIMA driver forecasting method with automatic parameter selection.

    Uses pmdarima.auto_arima to automatically find optimal ARIMA parameters:
    - Searches over multiple (p,d,q) combinations
    - Automatically detects and incorporates seasonality
    - Selects best model using AIC/BIC
    - More robust than fixed-parameter ARIMA

    Advantages over fixed ARIMA:
    - Optimized parameters per driver
    - Better handling of different data patterns
    - Automatic model selection
    - No manual parameter tuning needed

    Disadvantages:
    - Slower (tests multiple models)
    - Can overfit on small datasets
    - Less interpretable

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.AutoARIMADriverForecastParams
        Parameters specific to the Auto-ARIMA method.
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
        method_params: params.forecasting_params.AutoARIMADriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        super().__init__(
            driver_info, method_params, training_date, forecast_daterange, n_lag
        )

        # Extract Auto-ARIMA search parameters from method_params
        self.seasonal = method_params.seasonal
        self.m = method_params.m
        self.max_p = method_params.max_p
        self.max_q = method_params.max_q
        self.max_d = method_params.max_d
        self.max_P = method_params.max_P
        self.max_Q = method_params.max_Q
        self.max_D = method_params.max_D
        self.stepwise = method_params.stepwise
        self.suppress_warnings = method_params.suppress_warnings

        # Model will be trained later
        self.fitted_model: Any | None = None
        self.selected_order: tuple[int, int, int] | None = None
        self.selected_seasonal_order: tuple[int, int, int, int] | None = None

    @staticmethod
    def name() -> str:
        return 'Auto-ARIMA'

    def train(self) -> None:
        """
        Fit Auto-ARIMA model with automatic parameter selection.

        Searches over (p,d,q) parameter space to find optimal ARIMA configuration.
        Falls back to ARIMA(1,1,1) if automatic selection fails.
        """
        if not self.need_forecast():
            return

        training_data = self.get_training_data()

        # Search for optimal ARIMA parameters using AIC/BIC
        try:
            self.fitted_model = auto_arima(
                training_data,
                seasonal=self.seasonal,
                m=self.m,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                max_P=self.max_P if self.seasonal else 0,
                max_Q=self.max_Q if self.seasonal else 0,
                max_D=self.max_D if self.seasonal else 0,
                stepwise=self.stepwise,
                suppress_warnings=self.suppress_warnings,
                error_action='ignore',
                trace=False,
            )

            # Store selected parameters for logging
            # self.selected_order = self.fitted_model.order
            # self.selected_seasonal_order = self.fitted_model.seasonal_order

            # Log selected model
            # if self.seasonal and self.selected_seasonal_order != (0, 0, 0, 0):
            #     print(f"Auto-ARIMA: Selected SARIMA{self.selected_order}x{self.selected_seasonal_order} "
            #           f"for '{self.driver_info.name}'")
            # else:
            #     print(f"Auto-ARIMA: Selected ARIMA{self.selected_order} "
            #           f"for '{self.driver_info.name}' (no seasonality detected)")

            self.model = self.fitted_model

        except Exception as e:
            # Fall back to simple ARIMA(1,1,1) if search fails
            print(f"Auto-ARIMA failed for '{self.driver_info.name}': {e}")
            print('Falling back to ARIMA(1,1,1)')

            self.fitted_model = auto_arima(
                training_data,
                start_p=1,
                start_q=1,
                max_p=1,
                max_q=1,
                d=1,
                seasonal=False,
                stepwise=False,
                suppress_warnings=self.suppress_warnings,
                error_action='ignore',
            )

            self.selected_order = (1, 1, 1)
            self.selected_seasonal_order = (0, 0, 0, 0)
            self.model = self.fitted_model

    def apply(self) -> ArrayF:
        """
        Generate forecasts using the fitted Auto-ARIMA model.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.

        Raises
        ------
        RuntimeError
            If the model has not been trained when forecasting is required.
        """
        # Get number of forecast values required
        n_forecast = self.n_forecast_values_required()

        # If no forecast needed, return empty array
        if n_forecast <= 0:
            return np.array([])

        if self.fitted_model is None:
            raise RuntimeError('Model must be trained before forecasting')

        # Generate forecasts using selected optimal ARIMA model
        forecast = self.fitted_model.predict(n_periods=n_forecast)
        return np.array(forecast)

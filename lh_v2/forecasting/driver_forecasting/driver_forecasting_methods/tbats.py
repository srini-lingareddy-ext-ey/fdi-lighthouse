import datetime
from typing import Any

import numpy as np
from tbats import TBATS

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractDriverForecastingMethod


class TBATSDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend, Seasonal)
    forecasting method for drivers with complex seasonal patterns.

    Effective for multiple seasonal patterns, non-integer seasonal periods, and data
    with changing variance.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.TBATSDriverForecastParams
        Parameters specific to the TBATS forecasting method.
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
        method_params: params.forecasting_params.TBATSDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        super().__init__(
            driver_info, method_params, training_date, forecast_daterange, n_lag
        )

        # Extract TBATS parameters from method_params
        self.use_box_cox = method_params.use_box_cox
        self.use_trend = method_params.use_trend
        self.use_damped_trend = method_params.use_damped_trend
        self.seasonal_periods = method_params.seasonal_periods
        self.use_arma_errors = method_params.use_arma_errors

        # Model will be trained later
        self.fitted_model: Any | None = None

    @staticmethod
    def name() -> str:
        return 'TBATS'

    def train(self) -> None:
        """
        Train the TBATS model on historical driver data.

        Configures TBATS with specified parameters (None values enable auto-selection)
        and fits to training data.

        Raises
        ------
        ValueError
            If training data is insufficient or invalid.
        """
        training_data = self.get_training_data()

        if len(training_data) < max(self.seasonal_periods) * 2:
            raise ValueError(
                f'Insufficient training data for TBATS. Need at least '
                f'{max(self.seasonal_periods) * 2} observations, got {len(training_data)}'
            )

        # Configure TBATS estimator with parameters (None = auto-select)
        estimator = TBATS(
            use_box_cox=self.use_box_cox,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            seasonal_periods=self.seasonal_periods,
            use_arma_errors=self.use_arma_errors,
            show_warnings=False,
        )

        self.fitted_model = estimator.fit(training_data)
        self.model = self.fitted_model

    def apply(self) -> ArrayF:
        """
        Generate forecasts using the trained TBATS model.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.

        Raises
        ------
        ValueError
            If the model has not been trained when forecasting is required.
        """
        if self.fitted_model is None:
            raise ValueError(
                'Model must be trained before forecasting. Call train() first.'
            )

        n_forecast = self.n_forecast_values_required()

        # Generate forecast using fitted TBATS model
        forecast_result = self.fitted_model.forecast(steps=n_forecast)

        # Extract forecast values (handle tuple return with confidence intervals)
        if isinstance(forecast_result, tuple):
            forecast_values = forecast_result[0]
        else:
            forecast_values = forecast_result

        return np.array(forecast_values, dtype=self.driver_info.np_dtype)

import datetime

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractDriverForecastingMethod


class CrostonDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    Croston's method for forecasting intermittent demand drivers.

    Designed for time series with sporadic demand (many zeros). Separately models
    demand size and demand interval, with forecast = demand_size / interval.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.CrostonDriverForecastParams
        Parameters specific to the Croston forecasting method.
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
        method_params: params.forecasting_params.CrostonDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        super().__init__(
            driver_info=driver_info,
            method_params=method_params,
            training_date=training_date,
            forecast_daterange=forecast_daterange,
            n_lag=n_lag,
        )

        # Extract Croston parameters from method_params
        self.alpha = method_params.alpha
        self.variant = method_params.variant.lower()

        # Validate parameters
        if not 0 < self.alpha <= 1:
            raise ValueError(f'Alpha must be between 0 and 1, got {self.alpha}')

        if self.variant not in ['classic', 'sba', 'tsb']:
            raise ValueError(
                f"Variant must be 'classic', 'sba', or 'tsb', got {self.variant}"
            )

        # Model parameters will be estimated during training
        self.demand_size_estimate: float | None = None
        self.interval_estimate: float | None = None
        self.periods_since_last_demand: int = 0

    @staticmethod
    def name() -> str:
        return 'Croston'

    def train(self) -> None:
        """
        Train the Croston model by estimating demand size and interval parameters.

        Uses exponential smoothing to estimate average non-zero demand size and
        average time between demands.

        Raises
        ------
        ValueError
            If training data has insufficient non-zero demands (need at least 2).
        """
        training_data = self.get_training_data()

        # Identify non-zero demand occurrences
        non_zero_indices = np.where(training_data > 0)[0]

        if len(non_zero_indices) < 2:
            raise ValueError(
                f"Insufficient non-zero demands for Croston's method. "
                f'Found {len(non_zero_indices)}, need at least 2.'
            )

        # Initialize estimates with first non-zero demand
        first_demand_idx = non_zero_indices[0]
        self.demand_size_estimate = float(training_data[first_demand_idx])
        self.interval_estimate = float(first_demand_idx + 1)

        periods_since_last = 0

        # Update estimates using exponential smoothing at each non-zero demand
        for t in range(first_demand_idx + 1, len(training_data)):
            periods_since_last += 1

            if training_data[t] > 0:
                self.demand_size_estimate = float(
                    self.alpha * training_data[t]
                    + (1 - self.alpha) * self.demand_size_estimate
                )

                self.interval_estimate = float(
                    self.alpha * periods_since_last
                    + (1 - self.alpha) * self.interval_estimate
                )

                periods_since_last = 0

        self.periods_since_last_demand = periods_since_last
        self.model = True

    def apply(self) -> ArrayF:
        """
        Generate forecasts using the trained Croston model.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.
            Croston produces constant (flat) forecasts.

        Raises
        ------
        ValueError
            If the model has not been trained when forecasting is required.
        """
        if self.demand_size_estimate is None or self.interval_estimate is None:
            raise ValueError(
                'Model must be trained before forecasting. Call train() first.'
            )

        # Calculate forecast value based on selected variant
        if self.variant == 'classic':
            forecast_value = self.demand_size_estimate / self.interval_estimate

        elif self.variant == 'sba':
            # Syntetos-Boylan Approximation (bias-corrected)
            forecast_value = (
                (1 - self.alpha / 2)
                * self.demand_size_estimate
                / self.interval_estimate
            )

        elif self.variant == 'tsb':
            # Teunter-Syntetos-Babai (probability-weighted)
            probability = 1 / self.interval_estimate
            forecast_value = probability * self.demand_size_estimate
        else:
            raise ValueError(f'Invalid variant: {self.variant}')

        # Generate flat forecast array for all required periods
        n_forecast_periods = self.n_forecast_values_required()
        return np.full(n_forecast_periods, forecast_value, dtype=np.float64)

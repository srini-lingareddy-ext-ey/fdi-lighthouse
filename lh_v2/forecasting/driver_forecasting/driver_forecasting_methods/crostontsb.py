import datetime

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractDriverForecastingMethod


class CrostonTSBDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    Croston's TSB (Teunter-Syntetos-Babai) method for forecasting intermittent demand.

    Uses probability-based approach estimating demand occurrence probability rather than
    intervals. Forecast = probability × demand_size. Better for very intermittent demand.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.CrostonTSBDriverForecastParams
        Parameters specific to the Croston-TSB forecasting method.
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
        method_params: params.forecasting_params.CrostonTSBDriverForecastParams,
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

        # Extract Croston-TSB parameters from method_params
        self.alpha_demand = method_params.alpha_demand
        self.alpha_probability = method_params.alpha_probability

        if not 0 < self.alpha_demand <= 1:
            raise ValueError(
                f'alpha_demand must be between 0 and 1, got {self.alpha_demand}'
            )

        if not 0 < self.alpha_probability <= 1:
            raise ValueError(
                f'alpha_probability must be between 0 and 1, got {self.alpha_probability}'
            )

        # Model parameters will be estimated during training
        self.demand_size_estimate: float | None = None
        self.probability_estimate: float | None = None

    @staticmethod
    def name() -> str:
        return 'Croston-TSB'

    def train(self) -> None:
        """
        Train the Croston-TSB model by estimating demand size and probability.

        Uses exponential smoothing to estimate average non-zero demand size and
        probability of demand occurring in any given period.

        Raises
        ------
        ValueError
            If training data has insufficient non-zero demands (need at least 1).
        """
        training_data = self.get_training_data()

        # Identify non-zero demand occurrences
        non_zero_indices = np.where(training_data > 0)[0]

        if len(non_zero_indices) < 1:
            raise ValueError(
                f'Insufficient non-zero demands for Croston-TSB method. '
                f'Found {len(non_zero_indices)}, need at least 1.'
            )

        # Initialize estimates with first non-zero demand
        first_demand_idx = non_zero_indices[0]
        current_demand_size = float(training_data[first_demand_idx])
        current_probability = 1.0 / (first_demand_idx + 1)

        # Update estimates using exponential smoothing at each period
        for t in range(first_demand_idx + 1, len(training_data)):
            if training_data[t] > 0:
                # Demand occurred: update both demand size and probability (indicator = 1)
                current_demand_size = float(
                    self.alpha_demand * training_data[t]
                    + (1 - self.alpha_demand) * current_demand_size
                )

                current_probability = float(
                    self.alpha_probability * 1.0
                    + (1 - self.alpha_probability) * current_probability
                )
            else:
                # No demand: update only probability (indicator = 0)
                current_probability = float(
                    self.alpha_probability * 0.0
                    + (1 - self.alpha_probability) * current_probability
                )

        self.demand_size_estimate = current_demand_size
        self.probability_estimate = current_probability
        self.model = True

    def apply(self) -> ArrayF:
        """
        Generate forecasts using the trained Croston-TSB model.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.
            Croston-TSB produces constant (flat) forecasts.

        Raises
        ------
        ValueError
            If the model has not been trained when forecasting is required.
        """
        if self.demand_size_estimate is None or self.probability_estimate is None:
            raise ValueError(
                'Model must be trained before forecasting. Call train() first.'
            )

        # Calculate TSB forecast: probability × demand_size
        forecast_value = self.probability_estimate * self.demand_size_estimate

        # Generate flat forecast array for all required periods
        n_forecast_periods = self.n_forecast_values_required()
        return np.full(n_forecast_periods, forecast_value, dtype=np.float64)

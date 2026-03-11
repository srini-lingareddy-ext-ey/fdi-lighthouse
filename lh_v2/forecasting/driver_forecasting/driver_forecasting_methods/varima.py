import datetime
import warnings
from typing import Any

import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from .abstract_class import AbstractDriverForecastingMethod

logger = get_logger(__name__)


class VARIMADriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    VARIMA (Vector AutoRegressive Integrated Moving Average) forecasting method.

    Models multiple drivers simultaneously, capturing cross-correlations and
    interdependencies between related time series.

    Parameters
    ----------
    driver_info : dts.Driver
        Primary driver being forecasted.
    method_params : params.forecasting_params.VARIMADriverForecastParams
        Parameters specific to VARIMA forecasting.
    training_date : datetime.date | tuple[datetime.date, datetime.date]
        Training date or date range.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Forecast period start and end dates.
    n_lag : int
        Number of lag periods for this driver.
    additional_drivers : list[dts.Driver] | None
        Additional drivers to include in multivariate model.
    """

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.VARIMADriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
        additional_drivers: list[dts.Driver] | None = None,
    ):
        super().__init__(
            driver_info, method_params, training_date, forecast_daterange, n_lag
        )

        # Extract VARIMA parameters from method_params
        self.order = method_params.order
        self.trend = method_params.trend
        self.additional_drivers = additional_drivers or []

        # Model will be trained later
        self.fitted_model: Any | None = None
        self.all_drivers = [driver_info] + self.additional_drivers
        self.n_drivers = len(self.all_drivers)

    @staticmethod
    def name() -> str:
        return 'varima'

    def _prepare_multivariate_data(self) -> np.ndarray:
        """
        Prepare data matrix for VARIMA with shape (n_observations, n_drivers).

        Returns
        -------
        np.ndarray
            2D array where each column is a driver's time series.
        """
        # Get training data for primary driver
        primary_data = self.get_training_data()
        n_obs = len(primary_data)

        # Initialize data matrix
        data_matrix = np.zeros((n_obs, self.n_drivers))
        data_matrix[:, 0] = primary_data

        # Add additional drivers (aligned by date)
        for i, driver in enumerate(self.additional_drivers, start=1):
            # Get overlapping dates
            training_data = []

            if isinstance(self.training_date, tuple):
                start_date, end_date = self.training_date
            else:
                start_date = min(driver.dates.keys())
                end_date = self.training_date

            # Extract values for matching dates
            for date in sorted(driver.dates.keys()):
                if start_date <= date <= end_date:
                    idx = driver.dates[date]
                    training_data.append(driver.arr[idx])

            training_arr = np.array(training_data)

            # Verify alignment
            if len(training_arr) != n_obs:
                raise ValueError(
                    f"Driver '{driver.name}' has {len(training_arr)} observations, "
                    f'expected {n_obs}. Check date alignment.'
                )

            data_matrix[:, i] = training_arr

        return data_matrix

    def train(self) -> None:
        """
        Train VARIMA model on multiple drivers simultaneously.

        Models individual driver dynamics, cross-driver relationships, and moving
        average components. Falls back to mean forecast if training fails.
        """
        data = self._prepare_multivariate_data()

        if self.n_drivers == 1:
            logger.info(
                f"VARIMA: Only 1 driver provided, falling back to ARIMA-like behavior for '{self.driver_info.name}'"
            )
        else:
            driver_names = [d.name for d in self.all_drivers]
            logger.info(f'VARIMA: Training on {self.n_drivers} drivers: {driver_names}')

        # Fit VARMAX model with configured order and trend
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter('always')
                model = VARMAX(
                    endog=data,
                    order=self.order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                self.fitted_model = model.fit(disp=False, maxiter=200)
                self.model = self.fitted_model

            for w in caught_warnings:
                logger.warning(
                    f"VARIMA: Warning during training for '{self.driver_info.name}': {w.message}"
                )

            logger.info(f'VARIMA: Model trained successfully with order {self.order}')

        except Exception as e:
            logger.warning(
                f"VARIMA: Training failed for '{self.driver_info.name}': {e}"
            )
            logger.info('VARIMA: Falling back to mean forecast')
            self.fitted_model = None
            self.model = True

    def apply(self) -> ArrayF:
        """
        Generate forecasts for the primary driver using trained VARIMA model.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.
            Falls back to mean if training failed.
        """
        if not self.need_forecast():
            return np.array([])

        n_steps = self.n_forecast_values_required()

        # Fallback to mean if training failed
        if self.fitted_model is None:
            training_data = self.get_training_data()
            return np.full(n_steps, np.mean(training_data))

        try:
            # Generate multivariate forecast and extract primary driver (first column)
            forecast_result = self.fitted_model.forecast(steps=n_steps)

            if self.n_drivers == 1:
                primary_forecast = forecast_result.flatten()
            else:
                primary_forecast = forecast_result[:, 0]

            return primary_forecast.astype(self.driver_info.np_dtype)

        except Exception as e:
            logger.warning(
                f"VARIMA: Forecast failed for '{self.driver_info.name}': {e}"
            )
            training_data = self.get_training_data()
            return np.full(n_steps, np.mean(training_data))

import datetime
from typing import Optional

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import LinearRegressionAccountForecastParams
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class LinearRegressionDriversAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    Multiple Linear Regression forecasting method using selected drivers.

    Fits a linear regression model using drivers as features via ordinary
    least squares (OLS).

    Parameters
    ----------
    info : dts.AccountDriverGroup
        Account and driver information.
    best_lags : dict[dts.DriverName, int]
        Optimal lag for each driver.
    training_daterange : tuple[datetime.date, datetime.date]
        Training period start and end dates.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Forecast period start and end dates.
    model_params : LinearRegressionAccountForecastParams
        Parameters specific to Linear Regression forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: LinearRegressionAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, LinearRegressionAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{LinearRegressionAccountForecastParams}, Params were of type {type(model_params)}'
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

        # Model will be trained later (stores weights and intercept)
        self.model: tuple[ArrayF, float] | None = None

    @staticmethod
    def name() -> str:
        return 'Linear Regression with Drivers Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.LINEAR_REGRESSION_DRIVERS

    def train(self) -> None:
        """
        Train the multiple linear regression model using selected drivers.

        Fits OLS model: y = X @ weights + intercept.
        """
        training_info = self.get_training_data()

        X = training_info.drivers.arr.T
        y = training_info.account.arr

        # Add intercept column and solve via ordinary least squares
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

        intercept = coefficients[0]
        weights = coefficients[1:]

        self.model = (weights, intercept)

    def apply(self) -> ArrayF:
        """
        Apply the trained model to generate predictions.

        Returns
        -------
        ArrayF
            Predicted account values for the forecast period.

        Raises
        ------
        ModelNotTrainedError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ModelNotTrainedError(method_name=self.name())

        # Find available date range for each driver within forecast period
        start_dates = {}
        end_dates = {}

        for driver_name in self.info.drivers.get_ordered_drivers():
            driver = self.info.drivers.get_driver(driver_name)
            driver_dates = sorted(driver.dates.keys())

            # Find overlap between driver dates and forecast period
            forecast_start = self.forecast_daterange[0]
            forecast_end = self.forecast_daterange[1]

            # Get driver dates within forecast range
            available_dates = [
                d for d in driver_dates if forecast_start <= d <= forecast_end
            ]

            if not available_dates:
                raise ValueError(
                    f'Driver {driver_name} has no data in forecast period '
                    f'{forecast_start} to {forecast_end}. Driver data ends at {driver_dates[-1]}'
                )

            start_dates[driver_name] = available_dates[0]
            end_dates[driver_name] = available_dates[-1]

        forecast_driver_data = self.info.drivers.apply_daterange(
            start_dates=start_dates,
            end_dates=end_dates,
        )

        # Debug: Check the shape
        expected_months = (
            month_dif(self.forecast_daterange[0], self.forecast_daterange[1]) + 1
        )
        actual_samples = forecast_driver_data.arr.shape[1]

        if actual_samples != expected_months:
            raise ValueError(
                f'Driver data shape mismatch: expected {expected_months} months '
                f'from {self.forecast_daterange[0]} to {self.forecast_daterange[1]}, '
                f'but got {actual_samples} samples. '
                f'Start dates: {set(start_dates.values())}, End dates: {set(end_dates.values())}'
            )

        # Get driver data as features - shape: (n_drivers, n_samples)
        # Transpose to get (n_samples, n_drivers)
        X = forecast_driver_data.arr.T

        # Unpack model
        weights, intercept = self.model

        # Generate predictions: y = X @ weights + intercept
        predictions = X @ weights + intercept

        return predictions

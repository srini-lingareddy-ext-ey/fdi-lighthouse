import datetime
from typing import Optional

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
import lh_v2.stats as stats
from lh_v2.params.forecasting_params import LinearRegressionAccountForecastParams
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class LinearRegressionAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    Linear Regression forecasting method using time as the predictor.

    Fits a simple linear regression model with time index as the independent
    variable to predict account values.

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

        # Model will be trained later (stores slope and intercept)
        self.model: tuple[float, float] | None = None

    @staticmethod
    def name() -> str:
        return 'Linear Regression Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.LINEAR_REGRESSION

    @staticmethod
    def is_linear() -> bool:
        return True

    def train(self) -> None:
        """
        Train the simple linear regression model using time as predictor.

        Fits a line through historical account values using time index (0, 1, 2, ...)
        as the independent variable.
        """
        training_info = self.get_training_data()

        # Fit simple linear regression with time as predictor
        self.model = stats.slr(
            x=np.arange(
                month_dif(
                    start_date=min(training_info.account.dates.keys()),
                    end_date=max(training_info.account.dates.keys()),
                )
                + 1
            ).astype(float),
            y=training_info.account.arr,
        )

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

        training_info = self.get_training_data()

        training_length = (
            month_dif(
                start_date=min(training_info.account.dates.keys()),
                end_date=max(training_info.account.dates.keys()),
            )
            + 1
        )

        forecast_length = (
            month_dif(
                start_date=self.forecast_daterange[0],
                end_date=self.forecast_daterange[1],
            )
            + 1
        )

        # Generate time indices for forecast period continuing from training
        x_forecast = np.arange(
            training_length, training_length + forecast_length
        ).astype(float)

        predictions = stats.apply_slr(
            a=self.model[0],
            b=self.model[1],
            x=x_forecast,
        )
        return predictions

    def apply_vectorized(self, arr_input: ArrayF) -> ArrayF:
        """
        Apply the forecasting method to a vectorized input array.
        The array should be of shape (n_forecasts, n_features, n_samples)
        and the output should be of shape (n_forecasts, n_samples).
        For linear regression with time as predictor, we will ignore the features
        and just apply the same linear model to each sample.

        Parameters
        ----------
        arr_input : ArrayF
            Input array of shape (n_forecasts, n_features, n_samples)

        Returns
        -------
        ArrayF
            Output array of shape (n_forecasts, n_samples) with predictions.

        Raises
        ------
        ModelNotTrainedError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ModelNotTrainedError(method_name=self.name())

        n_forecasts, _, n_samples = arr_input.shape

        # Generate time indices for forecast period continuing from training
        training_info = self.get_training_data()
        training_length = (
            month_dif(
                start_date=min(training_info.account.dates.keys()),
                end_date=max(training_info.account.dates.keys()),
            )
            + 1
        )

        x_forecast = np.arange(training_length, training_length + n_forecasts).astype(
            float
        )

        # Apply the same linear model to each sample in the input array
        predictions = stats.apply_slr(
            a=self.model[0],
            b=self.model[1],
            x=x_forecast[:, np.newaxis],  # shape (n_forecasts, 1)
        )  # result will be shape (n_forecasts, 1)

        # Repeat predictions across samples to match output shape (n_forecasts, n_samples)
        predictions_repeated = np.repeat(predictions, repeats=n_samples, axis=1)

        return predictions_repeated

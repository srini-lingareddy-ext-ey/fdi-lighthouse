import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import ProphetAccountForecastParams
from lh_v2.shared import ArrayF

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class ProphetAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    Prophet-based forecasting method for account data.

    Designed for time series with strong seasonal patterns, multiple seasonality,
    and automatic changepoint detection. Robust to missing data and outliers.
    This is a time-series method that does not require driver data.

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
    model_params : ProphetAccountForecastParams
        Parameters specific to Prophet forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: ProphetAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, ProphetAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{ProphetAccountForecastParams}, Params were of type {type(model_params)}'
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

    @staticmethod
    def name() -> str:
        return 'Prophet Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.PROPHET

    def _prepare_training_dataframe(self) -> pd.DataFrame:
        """
        Prepare Prophet training dataframe with 'ds' (date) and 'y' (value) columns.

        Prophet requires a DataFrame with columns:
        - ds: datetime column
        - y: numeric column with values to forecast

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ds (datetime), y (float).
        """
        training_info = self.get_training_data()

        # Get account training data
        account_arr = training_info.account.arr

        # Extract dates for training period
        account_dates = sorted(training_info.account.dates.keys())

        # Create DataFrame in Prophet format
        df = pd.DataFrame({'ds': pd.to_datetime(account_dates), 'y': account_arr})

        return df

    def train(self) -> None:
        """
        Train Prophet model on historical account data.

        Prophet automatically detects trend changepoints and seasonal patterns.
        Suppresses verbose Prophet output during training.
        """
        # Prepare training data in Prophet format
        df = self._prepare_training_dataframe()

        try:
            # Initialize Prophet model with configured parameters
            self.fitted_model = Prophet(
                seasonality_mode=self.model_params.seasonality_mode,
                yearly_seasonality=self.model_params.yearly_seasonality,  # type: ignore[arg-type]
                weekly_seasonality=self.model_params.weekly_seasonality,  # type: ignore[arg-type]
                daily_seasonality=self.model_params.daily_seasonality,  # type: ignore[arg-type]
                changepoint_prior_scale=self.model_params.changepoint_prior_scale,
                seasonality_prior_scale=self.model_params.seasonality_prior_scale,
            )

            # Fit the model
            self.fitted_model.fit(df)
            self.model = self.fitted_model

        except Exception:
            # If fitting fails, set model to None
            self.fitted_model = None
            self.model = None

    def apply(self) -> ArrayF:
        """
        Generate forecasts using trained Prophet model.

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

        # Get forecast dates
        forecast_dates = self.get_forecasting_dates()

        # Create future dataframe for Prophet
        future_df = pd.DataFrame({'ds': pd.to_datetime(forecast_dates)})

        # Generate forecasts and extract point estimates (yhat)
        forecast = self.fitted_model.predict(future_df)
        forecast_values = forecast['yhat'].values

        return np.array(forecast_values, dtype=np.float64)

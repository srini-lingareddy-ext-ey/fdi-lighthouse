import datetime
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from .abstract_class import AbstractDriverForecastingMethod

logger = get_logger(__name__)


class ProphetDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    Prophet-based driver forecasting method.

    Designed for time series with strong seasonal patterns, multiple seasonality,
    and automatic changepoint detection. Robust to missing data and outliers.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.ProphetDriverForecastParams
        Parameters specific to Prophet forecasting.
    training_date : datetime.date | tuple[datetime.date, datetime.date]
        Training date or date range.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Forecast period start and end dates.
    n_lag : int
        Number of lag periods.
    additional_regressors : dict[str, ArrayF] | None
        Additional external regressors for the model.
    """

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.ProphetDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
        additional_regressors: dict[str, ArrayF] | None = None,
    ):
        super().__init__(
            driver_info, method_params, training_date, forecast_daterange, n_lag
        )

        # Extract Prophet parameters from method_params
        self.seasonality_mode = method_params.seasonality_mode
        self.yearly_seasonality = method_params.yearly_seasonality
        self.weekly_seasonality = method_params.weekly_seasonality
        self.daily_seasonality = method_params.daily_seasonality
        self.changepoint_prior_scale = method_params.changepoint_prior_scale
        self.seasonality_prior_scale = method_params.seasonality_prior_scale
        self.additional_regressors = additional_regressors or {}

        # Model will be trained later
        self.fitted_model: Any | None = None

    @staticmethod
    def name() -> str:
        return 'prophet'

    def _prepare_training_dataframe(self) -> pd.DataFrame:
        """
        Prepare Prophet training dataframe with 'ds' (date) and 'y' (value) columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ds (datetime), y (float), and any regressors.
        """
        training_data = self.get_training_data()

        # Get dates for training period
        if isinstance(self.training_date, tuple):
            start_date, end_date = self.training_date
        else:
            start_date = min(self.driver_info.dates.keys())
            end_date = self.training_date

        # Extract dates in training period
        dates = []
        for date in sorted(self.driver_info.dates.keys()):
            if start_date <= date <= end_date:
                dates.append(date)

        # Create DataFrame
        df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': training_data})

        # Add regressors if provided
        for regressor_name, regressor_values in self.additional_regressors.items():
            if len(regressor_values) != len(training_data):
                raise ValueError(
                    f"Regressor '{regressor_name}' has {len(regressor_values)} values, "
                    f'expected {len(training_data)}'
                )
            df[regressor_name] = regressor_values

        return df

    def train(self) -> None:
        """
        Train Prophet model on historical driver data.

        Prophet automatically detects trend changepoints and seasonal patterns.
        Falls back to None if fitting fails.
        """
        df = self._prepare_training_dataframe()

        # Initialize Prophet model with configured parameters
        self.fitted_model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,  # type: ignore[arg-type]
            weekly_seasonality=self.weekly_seasonality,  # type: ignore[arg-type]
            daily_seasonality=self.daily_seasonality,  # type: ignore[arg-type]
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )

        assert self.fitted_model is not None

        # Add any additional regressors
        for regressor_name in self.additional_regressors.keys():
            self.fitted_model.add_regressor(regressor_name)

        # Suppress Prophet's verbose output during fitting
        import logging

        logging.getLogger('prophet').setLevel(logging.WARNING)

        try:
            self.fitted_model.fit(df)
            self.model = self.fitted_model

            logger.info(f"Prophet: Trained successfully for '{self.driver_info.name}'")
            logger.info(f'  Seasonality mode: {self.seasonality_mode}')
            logger.info(f'  Yearly seasonality: {self.yearly_seasonality}')

        except Exception as e:
            logger.warning(
                f"Prophet: Training failed for '{self.driver_info.name}': {e}"
            )
            self.fitted_model = None
            self.model = None

    def apply(self) -> ArrayF:
        """
        Generate forecasts using trained Prophet model.

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
            # Generate future dates for forecast period
            from dateutil.relativedelta import relativedelta

            last_train_date = max(self.driver_info.dates.keys())
            future_dates = []
            current_date = last_train_date

            for _ in range(n_steps):
                current_date = current_date + relativedelta(months=1)
                future_dates.append(current_date)

            future_df = pd.DataFrame({'ds': pd.to_datetime(future_dates)})

            # Add regressor values for forecast period if provided
            for regressor_name, regressor_values in self.additional_regressors.items():
                training_len = len(self.get_training_data())
                future_regressor_values = regressor_values[
                    training_len : training_len + n_steps
                ]
                future_df[regressor_name] = future_regressor_values

            # Generate forecasts and extract point estimates (yhat)
            forecast = self.fitted_model.predict(future_df)
            forecast_values = forecast['yhat'].values

            return np.array(forecast_values, dtype=self.driver_info.np_dtype)

        except Exception as e:
            logger.warning(
                f"Prophet: Forecast failed for '{self.driver_info.name}': {e}"
            )
            training_data = self.get_training_data()
            return np.full(n_steps, np.mean(training_data))

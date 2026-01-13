import datetime

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params as params
import lh_v2.stats as stats
from lh_v2.params.forecasting_params import LinearRegressionDriverForecastParams
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif

from .abstract_class import AbstractDriverForecastingMethod


class LinearRegressionDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    Linear regression-based driver forecasting method.

    Uses simple linear regression to model trends in historical data and extrapolate
    into the forecast period.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.LinearRegressionDriverForecastParams
        Parameters specific to the linear regression forecasting method.
    training_date : datetime.date | tuple[datetime.date, datetime.date]
        Single date or date range (start, end) for training the model.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Date range (start, end) for which forecasts should be generated.
    n_lag : int
        Number of months to lag the forecast.
    """

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.LinearRegressionDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        # Validate params are correct type
        assert isinstance(method_params, LinearRegressionDriverForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{LinearRegressionDriverForecastParams}, Params were of type {type(method_params)}'
        )

        # Store forecasting configuration
        self.driver_info = driver_info
        self.method_params = method_params
        self.training_date = training_date
        self.forecast_daterange = forecast_daterange
        self.n_lag = n_lag

        # Model will be trained later
        self.model: tuple[float, float] | None = None

    @staticmethod
    def name() -> str:
        return 'Linear Regression Driver Forecasting Method'

    def train(self) -> None:
        """
        Train the linear regression model on the training data.

        Trains a simple linear regression model using sequential time indices as x
        and driver values as y. Training range is determined by training_date.
        """
        if not self.need_forecast():
            return

        training_data = self.get_training_data()

        # Determine training indices from training_date
        if isinstance(self.training_date, tuple):
            training_inds = (
                self.driver_info.dates[self.training_date[0]],
                self.driver_info.dates[self.training_date[1]],
            )
        else:
            training_inds = (0, self.driver_info.dates[self.training_date])

        # Fit linear model (y = mx + b) using sequential time indices
        self.model = stats.slr(
            x=np.arange(training_inds[0], training_inds[1] + 1).astype(float),
            y=training_data,
        )

    def apply(self) -> ArrayF:
        """
        Apply the linear regression model to generate forecasted values.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.

        Raises
        ------
        ValueError
            If the model has not been trained when forecasting is required.
        """
        if not self.need_forecast():
            return np.ndarray((0,), dtype=self.driver_info.np_dtype)

        if self.model is None:
            raise ValueError(f'Model not trained for method {self.name()}.')

        # Calculate time indices for forecast dates relative to min date
        min_date = min(self.driver_info.dates.keys())
        time_indices = np.array(
            [month_dif(min_date, date) for date in self.forecast_dates_required()]
        )

        # Apply linear model (y = mx + b) to forecast future values
        return stats.apply_slr(
            a=self.model[0],
            b=self.model[1],
            x=time_indices,
        )

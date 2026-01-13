import datetime
from typing import Optional

from sklearn.linear_model import Lasso

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import LassoAccountForecastParams
from lh_v2.shared import ArrayF

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class LassoAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    Lasso Regression forecasting method using selected drivers.

    Fits a linear regression model with L1 regularization for automatic
    feature selection.

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
    model_params : LassoAccountForecastParams
        Parameters specific to Lasso forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: LassoAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, LassoAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{LassoAccountForecastParams}, Params were of type {type(model_params)}'
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
        self.model: Lasso | None = None

    @staticmethod
    def name() -> str:
        return 'Lasso Regression Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.LASSO

    def train(self) -> None:
        """
        Train the Lasso regression model using selected drivers.

        Fits L1 regularized linear regression for automatic feature selection.
        """
        training_info = self.get_training_data()

        X = training_info.drivers.arr.T
        y = training_info.account.arr

        # Train Lasso with L1 regularization
        self.model = Lasso(
            alpha=self.model_params.alpha,
            fit_intercept=self.model_params.fit_intercept,
            max_iter=self.model_params.max_iter,
            tol=self.model_params.tol,
            selection=self.model_params.selection,
            random_state=self.model_params.random_state,
        )

        self.model.fit(X, y)

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

        # Get forecasted driver data for the forecast period
        forecasting_data = self.get_forecasting_input_data()
        forecasting_data = forecasting_data.drivers.apply_daterange_lags(
            start_date=self.forecast_daterange[0],
            end_date=self.forecast_daterange[1],
            lags=self.best_lags,
            b_training=False,
        )
        X = forecasting_data.arr.T
        predictions = self.model.predict(X)

        return predictions

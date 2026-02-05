import datetime
import warnings
from typing import Optional

from sklearn.linear_model import Ridge

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import RidgeAccountForecastParams
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod

logger = get_logger(__name__)


class RidgeAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    Ridge Regression forecasting method using selected drivers.

    Fits a linear regression model with L2 regularization to prevent overfitting
    and handle multicollinearity.

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
    model_params : RidgeAccountForecastParams
        Parameters specific to Ridge forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: RidgeAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, RidgeAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{RidgeAccountForecastParams}, Params were of type {type(model_params)}'
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
        self.model: Ridge | None = None

    @staticmethod
    def name() -> str:
        return 'Ridge Regression Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.RIDGE

    def train(self) -> None:
        """
        Train the Ridge regression model using selected drivers.

        Fits L2 regularized linear regression using driver data as features.
        """
        training_info = self.get_training_data()

        X = training_info.drivers.arr.T
        y = training_info.account.arr

        # Train Ridge with L2 regularization
        self.model = Ridge(
            alpha=self.model_params.alpha,
            fit_intercept=self.model_params.fit_intercept,
            solver=self.model_params.solver,
            random_state=self.model_params.random_state,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.model.fit(X, y)

            for warning in w:
                logger.warning(
                    f'{self.name()} encountered warning during training '
                    f'- {warning.category.__name__}: {warning.message}'
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

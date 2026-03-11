import datetime
from typing import Optional

from xgboost import XGBRegressor

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import XGBoostAccountForecastParams
from lh_v2.shared import ArrayF

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class XGBoostAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    XGBoost Regression forecasting method using selected drivers.

    Fits a gradient boosting ensemble that builds trees sequentially, with each
    tree correcting errors from previous trees.

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
    model_params : XGBoostAccountForecastParams
        Parameters specific to XGBoost forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: XGBoostAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, XGBoostAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{XGBoostAccountForecastParams}, Params were of type {type(model_params)}'
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
        self.model: XGBRegressor | None = None

    @staticmethod
    def name() -> str:
        return 'XGBoost Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.XGBOOST

    @staticmethod
    def is_linear() -> bool:
        return False

    def train(self) -> None:
        """
        Train the XGBoost regression model using selected drivers.

        Fits gradient boosting ensemble that builds trees sequentially to minimize
        prediction error.
        """
        training_info = self.get_training_data()

        X = training_info.drivers.arr.T
        y = training_info.account.arr

        # Train XGBoost with configured parameters
        self.model = XGBRegressor(
            n_estimators=self.model_params.n_estimators,
            max_depth=self.model_params.max_depth,
            learning_rate=self.model_params.learning_rate,
            subsample=self.model_params.subsample,
            colsample_bytree=self.model_params.colsample_bytree,
            reg_alpha=self.model_params.reg_alpha,
            reg_lambda=self.model_params.reg_lambda,
            random_state=self.model_params.random_state,
            n_jobs=self.model_params.n_jobs,
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

    def apply_vectorized(self, arr_input: ArrayF) -> ArrayF:
        if self.model is None:
            raise ModelNotTrainedError(method_name=self.name())

        new_shape = (arr_input.shape[0] * arr_input.shape[2], arr_input.shape[1])
        out_shape = (arr_input.shape[0], arr_input.shape[2])
        arr_flattened = arr_input.swapaxes(1, 2).reshape(new_shape)

        arr_predict: ArrayF = self.model.predict(arr_flattened)

        return arr_predict.reshape(out_shape)

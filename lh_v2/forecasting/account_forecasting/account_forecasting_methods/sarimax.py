import datetime
from itertools import product
from typing import Any, Optional

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params.forecasting_params import SARIMAXAccountForecastParams
from lh_v2.shared import ArrayF

from ..errors import ModelNotTrainedError
from .abstract_class import AbstractAccountForecastingMethod


class SARIMAXAccountForecastingMethod(AbstractAccountForecastingMethod):
    """
    SARIMAX forecasting method using selected drivers as exogenous variables.

    Fits a Seasonal AutoRegressive Integrated Moving Average model with drivers
    as exogenous variables to capture AR patterns, MA patterns, seasonality, and
    external influences.

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
    model_params : SARIMAXAccountForecastParams
        Parameters specific to SARIMAX forecasting.
    """

    def __init__(
        self,
        info: dts.AccountDriverGroup,
        best_lags: dict[dts.DriverName, int],
        training_daterange: tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        model_params: SARIMAXAccountForecastParams,
    ):
        # Validate params are correct type
        assert isinstance(model_params, SARIMAXAccountForecastParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{SARIMAXAccountForecastParams}, Params were of type {type(model_params)}'
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

    @staticmethod
    def name() -> str:
        return 'SARIMAX Account Forecasting Method'

    @staticmethod
    def method_enum() -> aft.AccountForecastingMethodEnum:
        return aft.AccountForecastingMethodEnum.SARIMAX

    def train(self) -> None:
        """
        Train the SARIMAX model using account history and driver data.

        Optionally searches for optimal (p,d,q)(P,D,Q,s) parameters via grid search.
        """
        training_info = self.get_training_data()

        endog = training_info.account.arr
        exog = training_info.drivers.arr.T

        if self.model_params.auto_params:
            # Auto-search for best parameters using AIC
            best_aic = np.inf
            best_params = None
            best_seasonal_params = None

            max_p, max_d, max_q, max_P, max_D, max_Q, s_period = (
                self.model_params.max_params
            )

            all_params_options: list[list[int]] = []
            all_params_options.append(list(range(max_p + 1)))
            all_params_options.append(list(range(max_d + 1)))
            all_params_options.append(list(range(max_q + 1)))
            all_params_options.append(list(range(max_P + 1)))
            all_params_options.append(list(range(max_D + 1)))
            all_params_options.append(list(range(max_Q + 1)))

            all_params_combos = list(product(*all_params_options))

            for params in all_params_combos:
                p_val, d_val, q_val, P_val, D_val, Q_val = params
                try:
                    model = SARIMAX(
                        endog,
                        exog=exog,
                        order=(p_val, d_val, q_val),
                        seasonal_order=(
                            P_val,
                            D_val,
                            Q_val,
                            s_period,
                        )
                        if self.model_params.seasonal
                        else (0, 0, 0, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    results = model.fit(disp=False)

                    if results.aic < best_aic:  # type: ignore[attr-defined]
                        best_aic = results.aic  # type: ignore[attr-defined]
                        best_params = (p_val, d_val, q_val)
                        best_seasonal_params = (
                            (P_val, D_val, Q_val, s_period)
                            if self.model_params.seasonal
                            else (0, 0, 0, 0)
                        )
                except Exception:
                    continue

            # Fit final model with best parameters
            if best_params is not None:
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=best_params,
                    seasonal_order=best_seasonal_params,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.model = model.fit(disp=False)
            else:
                # Fallback to default if search failed
                p_val, d_val, q_val, P_val, D_val, Q_val, s_period = (
                    self.model_params.default_params
                )
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=(p_val, d_val, q_val),
                    seasonal_order=(P_val, D_val, Q_val, s_period)
                    if self.model_params.seasonal
                    else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.model = model.fit(disp=False)
        else:
            # Use default parameters from config
            p_val, d_val, q_val, P_val, D_val, Q_val, s_period = (
                self.model_params.default_params
            )
            model = SARIMAX(
                endog,
                exog=exog,
                order=(p_val, d_val, q_val),
                seasonal_order=(P_val, D_val, Q_val, s_period)
                if self.model_params.seasonal
                else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model = model.fit(disp=False)

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

        # Get forecasted driver data as exogenous variables for forecast period
        forecasting_data = self.get_forecasting_input_data()
        forecasting_data = forecasting_data.drivers.apply_daterange_lags(
            start_date=self.forecast_daterange[0],
            end_date=self.forecast_daterange[1],
            lags=self.best_lags,
            b_training=False,
        )
        exog = forecasting_data.arr.T
        n_steps = exog.shape[0]

        predictions = self.model.forecast(steps=n_steps, exog=exog)

        # Convert to numpy array if pandas Series
        if hasattr(predictions, 'values'):
            return predictions.values
        return predictions

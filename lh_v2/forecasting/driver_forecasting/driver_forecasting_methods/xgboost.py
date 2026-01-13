import datetime

import numpy as np
import xgboost as xgb

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractDriverForecastingMethod


class XGBoostDriverForecastingMethod(AbstractDriverForecastingMethod):
    """
    XGBoost-based driver forecasting method with time series feature engineering.

    Creates lagged features, rolling statistics, and seasonal components to capture
    non-linear patterns in driver data.

    Parameters
    ----------
    driver_info : dts.Driver
        Driver information containing historical data and metadata.
    method_params : params.forecasting_params.XGBoostDriverForecastParams
        Parameters specific to the XGBoost forecasting method.
    training_date : datetime.date | tuple[datetime.date, datetime.date]
        Training date or date range.
    forecast_daterange : tuple[datetime.date, datetime.date]
        Forecast period start and end dates.
    n_lag : int
        Number of lag periods.
    """

    def __init__(
        self,
        driver_info: dts.Driver,
        method_params: params.forecasting_params.XGBoostDriverForecastParams,
        training_date: datetime.date | tuple[datetime.date, datetime.date],
        forecast_daterange: tuple[datetime.date, datetime.date],
        n_lag: int,
    ):
        super().__init__(
            driver_info, method_params, training_date, forecast_daterange, n_lag
        )

        # Extract XGBoost parameters from method_params
        self.n_lags = method_params.n_lags
        self.max_depth = method_params.max_depth
        self.learning_rate = method_params.learning_rate
        self.n_estimators = method_params.n_estimators

        # Model will be trained later
        self.fitted_model = None
        self.feature_scaler_mean = None
        self.feature_scaler_std = None

    @staticmethod
    def name() -> str:
        return 'XGBoost'

    def _create_features(
        self, data: ArrayF, start_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create time series features from raw data.

        Parameters
        ----------
        data : ArrayF
            Raw time series data.
        start_idx : int
            Starting index for time trend feature.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (features, targets) where features have shape (n_samples, n_features).
        """
        n = len(data)

        # Need at least n_lags + 1 points to create features
        if n <= self.n_lags:
            raise ValueError(f'Not enough data points ({n}) for {self.n_lags} lags')

        features_list = []
        targets = []

        # Create features for each time point (starting after n_lags)
        for i in range(self.n_lags, n):
            feature_vec = []

            # Lagged values
            for lag in range(1, self.n_lags + 1):
                feature_vec.append(data[i - lag])

            # Rolling statistics
            # 3-month window
            if i >= 3:
                feature_vec.append(np.mean(data[i - 3 : i]))
                feature_vec.append(np.std(data[i - 3 : i]))
            else:
                feature_vec.extend([data[i - 1], 0.0])

            # 6-month window
            if i >= 6:
                feature_vec.append(np.mean(data[i - 6 : i]))
                feature_vec.append(np.std(data[i - 6 : i]))
            else:
                feature_vec.extend([data[i - 1], 0.0])

            # 12-month window
            if i >= 12:
                feature_vec.append(np.mean(data[i - 12 : i]))
                feature_vec.append(np.std(data[i - 12 : i]))
            else:
                feature_vec.extend([data[i - 1], 0.0])

            # Time trend
            feature_vec.append(start_idx + i)

            # Month encoding (cyclical - sin/cos for month of year)
            month = (start_idx + i) % 12
            feature_vec.append(np.sin(2 * np.pi * month / 12))
            feature_vec.append(np.cos(2 * np.pi * month / 12))

            features_list.append(feature_vec)
            targets.append(data[i])

        features = np.array(features_list)
        targets = np.array(targets)

        return features, targets

    def train(self) -> None:
        """
        Fit XGBoost model to training data with engineered features.

        Creates lagged features, rolling statistics, and seasonal components,
        then trains gradient boosting model.
        """
        if not self.need_forecast():
            return

        training_data = self.get_training_data()

        # Create features from time series (lags, rolling stats, seasonality)
        X_train, y_train = self._create_features(training_data)

        # Normalize features for gradient boosting stability
        self.feature_scaler_mean = np.mean(X_train, axis=0)
        self.feature_scaler_std = np.std(X_train, axis=0) + 1e-8
        X_train_scaled = (X_train - self.feature_scaler_mean) / self.feature_scaler_std

        # Train XGBoost regressor with configured parameters
        self.fitted_model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )

        self.fitted_model.fit(X_train_scaled, y_train)
        self.model = self.fitted_model

    def apply(self) -> ArrayF:
        """
        Generate forecasts using the fitted XGBoost model.

        Returns
        -------
        ArrayF
            Forecasted driver values for the required forecast dates.
        """
        n_forecast = self.n_forecast_values_required()

        if n_forecast <= 0:
            return np.array([])

        if self.fitted_model is None:
            raise RuntimeError('Model must be trained before forecasting')

        training_data = self.get_training_data()
        history = training_data.copy()
        forecasts = []

        # Recursive forecasting: predict one step, add to history, repeat
        for step in range(n_forecast):
            recent_history = history[-(self.n_lags + 12) :]
            X_test, _ = self._create_features(
                recent_history,
                start_idx=len(training_data) - len(recent_history) + step,
            )

            X_test = X_test[-1:, :]
            X_test_scaled = (
                X_test - self.feature_scaler_mean
            ) / self.feature_scaler_std

            next_value = self.fitted_model.predict(X_test_scaled)[0]

            history = np.append(history, next_value)
            forecasts.append(next_value)

        return np.array(forecasts)

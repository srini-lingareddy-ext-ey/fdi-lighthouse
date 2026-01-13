from enum import Enum


class DriverForecastingMethodEnum(str, Enum):
    """
    Enumeration for Driver Forecasting Methods.

    This enumeration defines the methods available for forecasting driver data.

    Attributes
    ----------
    LINEAR_REGRESSION : str
        Represents the Linear Regression method for forecasting.
    ARIMA : str
        Represents the ARIMA/SARIMA method for forecasting with automatic seasonality detection.
    AUTO_ARIMA : str
        Represents the Auto-ARIMA method with automatic parameter optimization using pmdarima.
    EXPONENTIAL_SMOOTHING : str
        Represents the Exponential Smoothing (Holt-Winters) method for forecasting.
    XGBOOST : str
        Represents the XGBoost method for forecasting with time series feature engineering.
    VARIMA : str
        Represents the VARIMA (Vector ARIMA) method for multivariate forecasting that
        captures cross-correlations between drivers.
    PROPHET : str
        Represents the Prophet method for forecasting, suitable for handling seasonality.
    TBATS : str
        Represents the TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors,
        Trend, Seasonal) method for complex seasonal patterns.
    CROSTON : str
        Represents Croston's method for forecasting intermittent demand with many zero values.
    CROSTON_TSB : str
        Represents Croston's TSB (Teunter-Syntetos-Babai) variant using probability-based forecasting
        for very intermittent demand.
    """

    LINEAR_REGRESSION = 'linear_regression'
    MOVING_AVERAGE = 'moving_average'
    ARIMA = 'arima'
    AUTO_ARIMA = 'auto_arima'
    EXPONENTIAL_SMOOTHING = 'exponential_smoothing'
    XGBOOST = 'xgboost'
    VARIMA = 'varima'
    PROPHET = 'prophet'
    TBATS = 'tbats'
    CROSTON = 'croston'
    CROSTON_TSB = 'croston_tsb'

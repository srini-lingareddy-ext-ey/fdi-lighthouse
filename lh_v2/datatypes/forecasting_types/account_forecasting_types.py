from enum import Enum


class AccountForecastingMethodEnum(str, Enum):
    """
    Enumeration for Account Forecasting Methods.

    This enumeration defines the methods available for forecasting account data.

    Attributes
    ----------
    LINEAR_REGRESSION : str
        Represents the Linear Regression method for forecasting (time-based only).
    LINEAR_REGRESSION_DRIVERS : str
        Represents the Linear Regression method using selected drivers.
    RANDOM_FOREST : str
        Represents the Random Forest Regression method using selected drivers.
    XGBOOST : str
        Represents the XGBoost Regression method using selected drivers.
    RIDGE : str
        Represents the Ridge Regression method using selected drivers.
    LASSO : str
        Represents the Lasso Regression method using selected drivers with L1 regularization.
    HYPERLASSO : str
        Represents the HyperLasso Regression method with automatic alpha tuning via cross-validation.
    SARIMAX : str
        Represents the SARIMAX method for forecasting (time-based only).
    MOVING_AVERAGE : str
        Represents the Moving Average method for forecasting (time-based only).
    EXPONENTIAL_SMOOTHING : str
        Represents the Exponential Smoothing (Holt-Winters) method for forecasting (time-based only).
    PROPHET : str
        Represents the Prophet method for forecasting with automatic seasonality detection (time-based only).
    """

    LINEAR_REGRESSION = 'linear_regression'
    LINEAR_REGRESSION_DRIVERS = 'lr_drivers'
    RANDOM_FOREST = 'random_forest'
    XGBOOST = 'xgboost'
    RIDGE = 'ridge'
    LASSO = 'lasso'
    HYPERLASSO = 'hyperlasso'
    SARIMAX = 'sarimax'
    MOVING_AVERAGE = 'moving_average'
    EXPONENTIAL_SMOOTHING = 'exponential_smoothing'
    PROPHET = 'prophet'

    def is_nonlinear(self) -> bool:
        """Whether this model is nonlinear and requires sampling-based extrema estimation.

        Used by AUTO extrema estimation routing. Only tree-based models
        return True; everything else (linear, time-series-only) uses the
        cheaper correlation method.
        """
        return self in {
            AccountForecastingMethodEnum.RANDOM_FOREST,
            AccountForecastingMethodEnum.XGBOOST,
        }


class AccountValidationMetricEnum(str, Enum):
    """
    Enumeration for Account Validation Metrics.

    This enumeration defines the metrics used to validate forecasting models.

    Attributes
    ----------
    MSE_PERCENTAGE : str
        Represents the Mean Squared Error Percentage metric.
    RMSE_PERCENTAGE : str
        Represents the Root Mean Squared Error Percentage metric.
    MAPE : str
        Represents the Mean Absolute Percentage Error metric.
    STD : str
        Represents the Standard Deviation metric.
    """

    MSE_PERCENTAGE = 'MSE%'
    RMSE_PERCENTAGE = 'RMSE%'
    MAPE = 'MAPE'
    STD = 'STD'


class HyperparamOptMethodEnum(str, Enum):
    """
    Enumeration for Hyperparameter Optimization Methods.

    This enumeration defines the methods available for hyperparameter optimization.

    Attributes
    ----------
    GRID : str
        Represents the Grid Search optimization method.
    RANDOM : str
        Represents the Random Search optimization method.
    BAYESIAN : str
        Represents the Bayesian Optimization method.
    """

    GRID = 'grid'
    RANDOM = 'random'
    BAYESIAN = 'bayesian'

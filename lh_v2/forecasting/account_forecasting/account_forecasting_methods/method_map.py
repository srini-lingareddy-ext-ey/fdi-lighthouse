import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft

from .abstract_class import AbstractAccountForecastingMethod
from .exponential_smoothing import ExponentialSmoothingAccountForecastingMethod
from .hyperlasso import HyperLassoAccountForecastingMethod
from .lasso import LassoAccountForecastingMethod
from .linear_regression import LinearRegressionAccountForecastingMethod
from .lr_drivers import LinearRegressionDriversAccountForecastingMethod
from .moving_average import MovingAverageAccountForecastingMethod
from .prophet import ProphetAccountForecastingMethod
from .random_forest import RandomForestAccountForecastingMethod
from .ridge import RidgeAccountForecastingMethod
from .sarimax import SARIMAXAccountForecastingMethod
from .xgboost import XGBoostAccountForecastingMethod

ACCOUNT_FORECASTING_METHOD_MAP: dict[
    aft.AccountForecastingMethodEnum, type[AbstractAccountForecastingMethod]
] = {
    aft.AccountForecastingMethodEnum.LINEAR_REGRESSION: LinearRegressionAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.LINEAR_REGRESSION_DRIVERS: LinearRegressionDriversAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.RANDOM_FOREST: RandomForestAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.RIDGE: RidgeAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.LASSO: LassoAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.HYPERLASSO: HyperLassoAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.SARIMAX: SARIMAXAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.XGBOOST: XGBoostAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.MOVING_AVERAGE: MovingAverageAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.EXPONENTIAL_SMOOTHING: ExponentialSmoothingAccountForecastingMethod,
    aft.AccountForecastingMethodEnum.PROPHET: ProphetAccountForecastingMethod,
}

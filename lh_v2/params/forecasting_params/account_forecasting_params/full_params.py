from typing import Sequence

from pydantic import Field

from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    AccountForecastingMethodEnum,
    HyperparamOptMethodEnum,
)
from lh_v2.util import BaseParamsModel

from .method_params import (
    BaseAccountForecastParams,
    BaseAccountForecastParamsRange,
    ExponentialSmoothingAccountForecastParams,
    ExponentialSmoothingAccountForecastParamsRange,
    HyperLassoAccountForecastParams,
    HyperLassoAccountForecastParamsRange,
    LassoAccountForecastParams,
    LassoAccountForecastParamsRange,
    LinearRegressionAccountForecastParams,
    LinearRegressionAccountForecastParamsRange,
    MovingAverageAccountForecastParams,
    MovingAverageAccountForecastParamsRange,
    ProphetAccountForecastParams,
    ProphetAccountForecastParamsRange,
    RandomForestAccountForecastParams,
    RandomForestAccountForecastParamsRange,
    RidgeAccountForecastParams,
    RidgeAccountForecastParamsRange,
    SARIMAXAccountForecastParams,
    SARIMAXAccountForecastParamsRange,
    XGBoostAccountForecastParams,
    XGBoostAccountForecastParamsRange,
)


class AccountForecastMethodParams(BaseParamsModel):
    b_remove: bool = True
    methods_removed: Sequence[AccountForecastingMethodEnum] = (
        AccountForecastingMethodEnum.LINEAR_REGRESSION_DRIVERS,
        AccountForecastingMethodEnum.SARIMAX,
        AccountForecastingMethodEnum.RIDGE,
    )
    b_selected: bool = False
    methods_selected: Sequence[AccountForecastingMethodEnum] = ()


class HyperparamOptParams(BaseParamsModel):
    b_optimize: bool = True
    method: HyperparamOptMethodEnum = HyperparamOptMethodEnum.BAYESIAN
    disabled_methods: Sequence[AccountForecastingMethodEnum] = (
        AccountForecastingMethodEnum.PROPHET,
    )
    n_options_per_param: int = 3
    # Multiplier for Bayesian optimization trials (n_options * n_params * multiplier)
    bayesian_trial_multiplier: int = 5
    linear_regression_param_range: LinearRegressionAccountForecastParamsRange = Field(
        default_factory=lambda: LinearRegressionAccountForecastParamsRange()
    )
    linear_regression_drivers_param_range: LinearRegressionAccountForecastParamsRange = Field(
        default_factory=lambda: LinearRegressionAccountForecastParamsRange()
    )
    random_forest_param_range: RandomForestAccountForecastParamsRange = Field(
        default_factory=lambda: RandomForestAccountForecastParamsRange()
    )
    xgboost_param_range: XGBoostAccountForecastParamsRange = Field(
        default_factory=lambda: XGBoostAccountForecastParamsRange()
    )
    ridge_param_range: RidgeAccountForecastParamsRange = Field(
        default_factory=lambda: RidgeAccountForecastParamsRange()
    )
    lasso_param_range: LassoAccountForecastParamsRange = Field(
        default_factory=lambda: LassoAccountForecastParamsRange()
    )
    hyperlasso_param_range: HyperLassoAccountForecastParamsRange = Field(
        default_factory=lambda: HyperLassoAccountForecastParamsRange()
    )
    sarimax_param_range: SARIMAXAccountForecastParamsRange = Field(
        default_factory=lambda: SARIMAXAccountForecastParamsRange()
    )
    moving_average_param_range: MovingAverageAccountForecastParamsRange = Field(
        default_factory=lambda: MovingAverageAccountForecastParamsRange()
    )
    exponential_smoothing_param_range: ExponentialSmoothingAccountForecastParamsRange = Field(
        default_factory=lambda: ExponentialSmoothingAccountForecastParamsRange()
    )
    prophet_param_range: ProphetAccountForecastParamsRange = Field(
        default_factory=lambda: ProphetAccountForecastParamsRange()
    )

    def __getitem__(
        self,
        key: AccountForecastingMethodEnum,
    ) -> BaseAccountForecastParamsRange:
        match key:
            case AccountForecastingMethodEnum.LINEAR_REGRESSION:
                return self.linear_regression_param_range
            case AccountForecastingMethodEnum.LINEAR_REGRESSION_DRIVERS:
                return self.linear_regression_drivers_param_range
            case AccountForecastingMethodEnum.RANDOM_FOREST:
                return self.random_forest_param_range
            case AccountForecastingMethodEnum.XGBOOST:
                return self.xgboost_param_range
            case AccountForecastingMethodEnum.RIDGE:
                return self.ridge_param_range
            case AccountForecastingMethodEnum.LASSO:
                return self.lasso_param_range
            case AccountForecastingMethodEnum.HYPERLASSO:
                return self.hyperlasso_param_range
            case AccountForecastingMethodEnum.SARIMAX:
                return self.sarimax_param_range
            case AccountForecastingMethodEnum.MOVING_AVERAGE:
                return self.moving_average_param_range
            case AccountForecastingMethodEnum.EXPONENTIAL_SMOOTHING:
                return self.exponential_smoothing_param_range
            case AccountForecastingMethodEnum.PROPHET:
                return self.prophet_param_range
            case _:
                raise ValueError(f'Invalid method: {key}')


class AccountForecastParams(BaseParamsModel):
    methods: AccountForecastMethodParams = Field(
        default_factory=lambda: AccountForecastMethodParams()
    )
    hyperparam_opt_params: HyperparamOptParams = Field(
        default_factory=lambda: HyperparamOptParams()
    )
    linear_regression_params: LinearRegressionAccountForecastParams = Field(
        default_factory=lambda: LinearRegressionAccountForecastParams()
    )
    linear_regression_drivers_params: LinearRegressionAccountForecastParams = Field(
        default_factory=lambda: LinearRegressionAccountForecastParams()
    )
    random_forest_params: RandomForestAccountForecastParams = Field(
        default_factory=lambda: RandomForestAccountForecastParams()
    )
    xgboost_params: XGBoostAccountForecastParams = Field(
        default_factory=lambda: XGBoostAccountForecastParams()
    )
    ridge_params: RidgeAccountForecastParams = Field(
        default_factory=lambda: RidgeAccountForecastParams()
    )
    lasso_params: LassoAccountForecastParams = Field(
        default_factory=lambda: LassoAccountForecastParams()
    )
    hyperlasso_params: HyperLassoAccountForecastParams = Field(
        default_factory=lambda: HyperLassoAccountForecastParams()
    )
    sarimax_params: SARIMAXAccountForecastParams = Field(
        default_factory=lambda: SARIMAXAccountForecastParams()
    )
    moving_average_params: MovingAverageAccountForecastParams = Field(
        default_factory=lambda: MovingAverageAccountForecastParams()
    )
    exponential_smoothing_params: ExponentialSmoothingAccountForecastParams = Field(
        default_factory=lambda: ExponentialSmoothingAccountForecastParams()
    )
    prophet_params: ProphetAccountForecastParams = Field(
        default_factory=lambda: ProphetAccountForecastParams()
    )

    def __getitem__(
        self, key: AccountForecastingMethodEnum
    ) -> BaseAccountForecastParams:
        match key:
            case AccountForecastingMethodEnum.LINEAR_REGRESSION:
                return self.linear_regression_params
            case AccountForecastingMethodEnum.LINEAR_REGRESSION_DRIVERS:
                return self.linear_regression_drivers_params
            case AccountForecastingMethodEnum.RANDOM_FOREST:
                return self.random_forest_params
            case AccountForecastingMethodEnum.XGBOOST:
                return self.xgboost_params
            case AccountForecastingMethodEnum.RIDGE:
                return self.ridge_params
            case AccountForecastingMethodEnum.LASSO:
                return self.lasso_params
            case AccountForecastingMethodEnum.HYPERLASSO:
                return self.hyperlasso_params
            case AccountForecastingMethodEnum.SARIMAX:
                return self.sarimax_params
            case AccountForecastingMethodEnum.MOVING_AVERAGE:
                return self.moving_average_params
            case AccountForecastingMethodEnum.EXPONENTIAL_SMOOTHING:
                return self.exponential_smoothing_params
            case AccountForecastingMethodEnum.PROPHET:
                return self.prophet_params
            case _:
                raise ValueError(f'Invalid method: {key}')

    def set_method_params(
        self, key: AccountForecastingMethodEnum, params: BaseAccountForecastParams
    ):
        match key:
            case AccountForecastingMethodEnum.LINEAR_REGRESSION:
                assert isinstance(params, LinearRegressionAccountForecastParams)
                self.linear_regression_params = params
            case AccountForecastingMethodEnum.LINEAR_REGRESSION_DRIVERS:
                assert isinstance(params, LinearRegressionAccountForecastParams)
                self.linear_regression_drivers_params = params
            case AccountForecastingMethodEnum.RANDOM_FOREST:
                assert isinstance(params, RandomForestAccountForecastParams)
                self.random_forest_params = params
            case AccountForecastingMethodEnum.XGBOOST:
                assert isinstance(params, XGBoostAccountForecastParams)
                self.xgboost_params = params
            case AccountForecastingMethodEnum.RIDGE:
                assert isinstance(params, RidgeAccountForecastParams)
                self.ridge_params = params
            case AccountForecastingMethodEnum.LASSO:
                assert isinstance(params, LassoAccountForecastParams)
                self.lasso_params = params
            case AccountForecastingMethodEnum.HYPERLASSO:
                assert isinstance(params, HyperLassoAccountForecastParams)
                self.hyperlasso_params = params
            case AccountForecastingMethodEnum.SARIMAX:
                assert isinstance(params, SARIMAXAccountForecastParams)
                self.sarimax_params = params
            case AccountForecastingMethodEnum.MOVING_AVERAGE:
                assert isinstance(params, MovingAverageAccountForecastParams)
                self.moving_average_params = params
            case AccountForecastingMethodEnum.EXPONENTIAL_SMOOTHING:
                assert isinstance(params, ExponentialSmoothingAccountForecastParams)
                self.exponential_smoothing_params = params
            case AccountForecastingMethodEnum.PROPHET:
                assert isinstance(params, ProphetAccountForecastParams)
                self.prophet_params = params
            case _:
                raise ValueError(f'Invalid method: {key}')

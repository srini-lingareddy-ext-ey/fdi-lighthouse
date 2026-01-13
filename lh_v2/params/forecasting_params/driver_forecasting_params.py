from typing import Literal

from pydantic import Field

from lh_v2.datatypes.forecasting_types.driver_forecasting_types import (
    DriverForecastingMethodEnum,
)
from lh_v2.util import BaseParamsModel


class BaseDriverForecastParams(BaseParamsModel):
    pass


class LinearRegressionDriverForecastParams(BaseDriverForecastParams):
    pass


class MovingAverageDriverForecastParams(BaseDriverForecastParams):
    window_size: int = 3  # Number of periods to average


class ARIMADriverForecastParams(BaseDriverForecastParams):
    order: tuple[int, int, int] = (1, 1, 1)  # (p, d, q) ARIMA parameters
    seasonal_order: tuple[int, int, int, int] | None = (
        None  # (P, D, Q, s) seasonal parameters
    )
    auto_seasonality: bool = True  # Automatically detect seasonality
    seasonality_threshold: float = 0.1  # Threshold for seasonal strength (10%)
    seasonal_period: int = 12  # Period for seasonality (12 for monthly data)


class AutoARIMADriverForecastParams(BaseDriverForecastParams):
    seasonal: bool = True  # Whether to fit seasonal ARIMA
    m: int = 12  # Seasonal period (12 for monthly data)
    max_p: int = 5  # Maximum p for auto search
    max_q: int = 5  # Maximum q for auto search
    max_P: int = 2  # Maximum seasonal P for auto search
    max_Q: int = 2  # Maximum seasonal Q for auto search
    max_d: int = 2  # Maximum d for auto search
    max_D: int = 1  # Maximum seasonal D for auto search
    stepwise: bool = True  # Use stepwise algorithm for faster search
    suppress_warnings: bool = True  # Suppress convergence warnings


class XGBoostDriverForecastParams(BaseDriverForecastParams):
    n_lags: int = 12  # Number of lagged features to use
    max_depth: int = 3  # Maximum tree depth
    learning_rate: float = 0.1  # Learning rate (eta)
    n_estimators: int = 100  # Number of boosting rounds


class TBATSDriverForecastParams(BaseDriverForecastParams):
    use_box_cox: bool | None = None  # Box-Cox transformation (None = auto-select)
    use_trend: bool | None = None  # Include trend component (None = auto-select)
    use_damped_trend: bool | None = None  # Use damped trend (None = auto-select)
    seasonal_periods: tuple[int, ...] = (12,)  # Seasonal periods to model
    use_arma_errors: bool = True  # Model residuals with ARMA


class CrostonDriverForecastParams(BaseDriverForecastParams):
    alpha: float = 0.1  # Smoothing parameter (0 < alpha <= 1)
    variant: Literal['classic', 'sba', 'tsb'] = (
        'classic'  # Croston variant: 'classic', 'sba', or 'tsb'
    )


class CrostonTSBDriverForecastParams(BaseDriverForecastParams):
    alpha_demand: float = 0.1  # Smoothing parameter for demand size
    alpha_probability: float = 0.1  # Smoothing parameter for demand probability


class VARIMADriverForecastParams(BaseDriverForecastParams):
    order: tuple[int, int] = (1, 1)  # (p, q) VAR and MA orders
    trend: Literal['n', 'c', 't', 'ct'] = (
        'ct'  # Trend: 'n'=none, 'c'=constant, 't'=linear, 'ct'=both
    )


class ExponentialSmoothingDriverForecastParams(BaseDriverForecastParams):
    trend: Literal['add', 'mul', None] | None = (
        None  # Trend type: 'add', 'mul', or None (None = auto-detect)
    )
    seasonal: Literal['add', 'mul', None] | None = (
        None  # Seasonal type: 'add', 'mul', or None (None = auto-detect)
    )
    seasonal_periods: int = 12  # Seasonal period (12 for monthly data)
    auto_detect: bool = True  # Auto-detect trend and seasonality
    seasonality_threshold: float = 0.1  # Threshold for seasonal strength (10%)


class ProphetDriverForecastParams(BaseDriverForecastParams):
    seasonality_mode: Literal['additive', 'multiplicative'] = (
        'additive'  # 'additive' or 'multiplicative'
    )
    yearly_seasonality: bool | int | str = (
        'auto'  # Yearly seasonality (auto/True/False/int)
    )
    weekly_seasonality: bool | str = (
        False  # Weekly seasonality (usually False for monthly)
    )
    daily_seasonality: bool | str = (
        False  # Daily seasonality (usually False for monthly)
    )
    changepoint_prior_scale: float = 0.05  # Trend flexibility (higher = more flexible)
    seasonality_prior_scale: float = (
        10.0  # Seasonality flexibility (higher = more flexible)
    )


class DriverForecastParams(BaseParamsModel):
    selected_method: DriverForecastingMethodEnum = DriverForecastingMethodEnum.XGBOOST
    linear_regression_params: LinearRegressionDriverForecastParams = Field(
        default_factory=lambda: LinearRegressionDriverForecastParams()
    )
    moving_average_params: MovingAverageDriverForecastParams = Field(
        default_factory=lambda: MovingAverageDriverForecastParams()
    )
    arima_params: ARIMADriverForecastParams = Field(
        default_factory=lambda: ARIMADriverForecastParams()
    )
    auto_arima_params: AutoARIMADriverForecastParams = Field(
        default_factory=lambda: AutoARIMADriverForecastParams()
    )
    xgboost_params: XGBoostDriverForecastParams = Field(
        default_factory=lambda: XGBoostDriverForecastParams()
    )
    tbats_params: TBATSDriverForecastParams = Field(
        default_factory=lambda: TBATSDriverForecastParams()
    )
    croston_params: CrostonDriverForecastParams = Field(
        default_factory=lambda: CrostonDriverForecastParams()
    )
    croston_tsb_params: CrostonTSBDriverForecastParams = Field(
        default_factory=lambda: CrostonTSBDriverForecastParams()
    )
    varima_params: VARIMADriverForecastParams = Field(
        default_factory=lambda: VARIMADriverForecastParams()
    )
    exponential_smoothing_params: ExponentialSmoothingDriverForecastParams = Field(
        default_factory=lambda: ExponentialSmoothingDriverForecastParams()
    )
    prophet_params: ProphetDriverForecastParams = Field(
        default_factory=lambda: ProphetDriverForecastParams()
    )

    def __getitem__(self, key: DriverForecastingMethodEnum) -> BaseDriverForecastParams:
        match key:
            case DriverForecastingMethodEnum.LINEAR_REGRESSION:
                return self.linear_regression_params
            case DriverForecastingMethodEnum.MOVING_AVERAGE:
                return self.moving_average_params
            case DriverForecastingMethodEnum.ARIMA:
                return self.arima_params
            case DriverForecastingMethodEnum.AUTO_ARIMA:
                return self.auto_arima_params
            case DriverForecastingMethodEnum.XGBOOST:
                return self.xgboost_params
            case DriverForecastingMethodEnum.TBATS:
                return self.tbats_params
            case DriverForecastingMethodEnum.CROSTON:
                return self.croston_params
            case DriverForecastingMethodEnum.CROSTON_TSB:
                return self.croston_tsb_params
            case DriverForecastingMethodEnum.VARIMA:
                return self.varima_params
            case DriverForecastingMethodEnum.EXPONENTIAL_SMOOTHING:
                return self.exponential_smoothing_params
            case DriverForecastingMethodEnum.PROPHET:
                return self.prophet_params
            case _:
                raise ValueError(
                    f"Driver forecasting method '{key}' not supported or not configured"
                )

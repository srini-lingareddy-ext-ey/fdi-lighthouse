from pydantic import Field

from lh_v2.datatypes.scenario_planning_types import (
    SPDriverForecastingMethodEnum,
)
from lh_v2.util import BaseParamsModel


class BaseSPDriverForecastingMethodParams(BaseParamsModel):
    pass


class SPLinearDriverForecastingParams(BaseSPDriverForecastingMethodParams):
    pass


class SPVARBasedDriverForecastingParams(BaseSPDriverForecastingMethodParams):
    multi_slr_seg_len: int = 8
    """
    The segment length for the piecewise linear 
    regression used in the VAR-based driver forecasting method.
    """
    time_scaling_factor: float = 1 / 6
    """
    The scaling factor applied to the time component 
    in the variance-based perturbation method.
    """
    seasonality_threshold: float = 0.1
    """
    Seasonal strength threshold (var(seasonal) / var(ts)).
    Drivers exceeding this are deseasonalized before computing σ.
    """
    period_checks: list[int] = Field(default_factory=lambda: [6, 12])
    """
    List of periods to check for seasonality in the driver time series.
    """


class SPEMADriverForecastingParams(BaseSPDriverForecastingMethodParams):
    a: float = 0.5
    """The alpha parameter for the EMA driver forecasting method."""


class SPSarimaxDriverForecastingParams(BaseSPDriverForecastingMethodParams):
    seasonal: bool = True
    """Whether to use seasonal components in the SARIMAX model."""
    alpha: float = 0.05
    """The significance level for the SARIMAX model."""


class SPDriverForecastingParams(BaseParamsModel):
    selected_perturbation_method: SPDriverForecastingMethodEnum = (
        SPDriverForecastingMethodEnum.VAR_BASED
    )
    """The selected method for driver perturbation."""
    linear_params: SPLinearDriverForecastingParams = Field(
        default_factory=SPLinearDriverForecastingParams
    )
    """Parameters for the linear driver forecasting method."""

    ema_params: SPEMADriverForecastingParams = Field(
        default_factory=SPEMADriverForecastingParams
    )
    """Parameters for the EMA driver forecasting method."""

    variance_based_params: SPVARBasedDriverForecastingParams = Field(
        default_factory=SPVARBasedDriverForecastingParams
    )
    """Parameters for the VAR-based driver forecasting method."""

    sarimax_params: SPSarimaxDriverForecastingParams = Field(
        default_factory=SPSarimaxDriverForecastingParams
    )
    """Parameters for the SARIMAX driver forecasting method."""

    def get_selected_perturbation_method_params(
        self,
    ) -> BaseSPDriverForecastingMethodParams:
        """
        Get the parameters for the selected driver forecasting method.

        Returns:
            BaseSPDriverForecastingMethodParams: The parameters for the selected method.
        """
        match self.selected_perturbation_method:
            case SPDriverForecastingMethodEnum.VAR_BASED:
                return self.variance_based_params
            case _:
                raise ValueError(
                    f'Unsupported forecasting method: {self.selected_perturbation_method}'
                )

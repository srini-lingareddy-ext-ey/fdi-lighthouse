from pydantic import Field

from lh_v2.util import BaseParamsModel

from .sp_driver_forecasting_params import SPDriverForecastingParams
from .sp_extrema_estimation_params import SPExtremaEstimationParams


class ScenarioPlanningParams(BaseParamsModel):
    b_scenario_plan: bool = True
    """Whether to perform scenario planning."""
    sp_driver_forecasting_params: SPDriverForecastingParams = Field(
        default_factory=SPDriverForecastingParams
    )
    """Parameters for driver forecasting in scenario planning."""
    sp_extrema_estimation_params: SPExtremaEstimationParams = Field(
        default_factory=SPExtremaEstimationParams
    )
    """Parameters for extrema estimation in scenario planning."""

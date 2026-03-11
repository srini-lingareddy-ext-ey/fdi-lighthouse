from pydantic import Field

from lh_v2.datatypes.scenario_planning_types import SPExtremaEstimationMethodEnum
from lh_v2.util import BaseParamsModel


class SPExtremaEstimationMethodParams(BaseParamsModel):
    pass


class SPAutoExtremaEstimationMethodParams(SPExtremaEstimationMethodParams):
    pass


class SPCorrelationExtremaEstimationMethodParams(SPExtremaEstimationMethodParams):
    pass


class SPSamplingExtremaEstimationMethodParams(SPExtremaEstimationMethodParams):
    n_samples: int = 500
    """
    Number of random driver-scenario combinations to evaluate when using
    the 'sampling' extrema estimation method. Clamped to
    [200, total_combinations] at runtime. Ignored for other methods.
    """


class SPExactExtremaEstimationMethodParams(SPExtremaEstimationMethodParams):
    pass


class SPExtremaEstimationParams(BaseParamsModel):
    extrema_estimation_method: SPExtremaEstimationMethodEnum = (
        SPExtremaEstimationMethodEnum.EXACT
    )

    auto_extrema_estimation_params: SPAutoExtremaEstimationMethodParams = Field(
        default_factory=SPAutoExtremaEstimationMethodParams
    )

    correlation_extrema_estimation_params: SPCorrelationExtremaEstimationMethodParams = Field(
        default_factory=SPCorrelationExtremaEstimationMethodParams
    )

    sampling_extrema_estimation_params: SPSamplingExtremaEstimationMethodParams = Field(
        default_factory=SPSamplingExtremaEstimationMethodParams
    )

    exact_extrema_estimation_params: SPExactExtremaEstimationMethodParams = Field(
        default_factory=SPExactExtremaEstimationMethodParams
    )

    def __getitem__(
        self, key: SPExtremaEstimationMethodEnum
    ) -> SPExtremaEstimationMethodParams:
        match key:
            case SPExtremaEstimationMethodEnum.AUTO:
                return self.auto_extrema_estimation_params
            case SPExtremaEstimationMethodEnum.CORRELATION:
                return self.correlation_extrema_estimation_params
            case SPExtremaEstimationMethodEnum.SAMPLING:
                return self.sampling_extrema_estimation_params
            case SPExtremaEstimationMethodEnum.EXACT:
                return self.exact_extrema_estimation_params
            case _:
                raise KeyError(f'Invalid extrema estimation method: {key}')

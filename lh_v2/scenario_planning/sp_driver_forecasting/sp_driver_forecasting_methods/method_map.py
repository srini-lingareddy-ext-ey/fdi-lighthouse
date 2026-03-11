from lh_v2.datatypes.scenario_planning_types import (
    SPDriverForecastingMethodEnum,
)

from .abstract_class import AbstractSPDriverPerturbationMethod
from .variance_based_perterbation import VarianceBasedDriverPerturbationMethod

SP_DRIVER_PERTURBATION_METHOD_MAP: dict[
    SPDriverForecastingMethodEnum, type[AbstractSPDriverPerturbationMethod]
] = {
    SPDriverForecastingMethodEnum.VAR_BASED: VarianceBasedDriverPerturbationMethod,
}

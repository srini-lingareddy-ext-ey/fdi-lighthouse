from dataclasses import dataclass

from lh_v2.datatypes import (
    AccountGroupInfo,
    AccountGroupSelectedDrivers,
    AccountType,
    DriverClassification,
    DriverName,
)
from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    AccountForecastingMethodEnum,
)
from lh_v2.datatypes.scenario_planning_types import (
    SPExtremaCase,
)
from lh_v2.params import AccountForecastParams


@dataclass
class ScenarioPlanningInput:
    """
    Input configuration for scenario planning of drivers.

    This class encapsulates all the necessary parameters and configurations
    required to perform scenario planning for different driver groups using various
    perturbation methods.
    """

    accounts_drivers: AccountGroupSelectedDrivers
    lags: dict[AccountType, dict[DriverName, int]]
    classifications: dict[AccountType, dict[DriverName, DriverClassification]]
    selected_model: dict[AccountType, AccountForecastingMethodEnum]
    best_params: dict[AccountType, AccountForecastParams]


@dataclass
class ScenarioPlanningOutput:
    perturbed_accounts: AccountGroupInfo
    extrema_accounts: dict[SPExtremaCase, AccountGroupInfo]

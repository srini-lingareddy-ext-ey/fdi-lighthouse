import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
    SPExtremaCase,
    SPExtremaEstimationMethodEnum,
)
from lh_v2.forecasting.account_forecasting.account_forecasting_methods import (
    AbstractAccountForecastingMethod,
)
from lh_v2.params import GeneralParams, SPExtremaEstimationParams

from .extrema_estimation_correlation import extrema_estimation_correlation_method
from .extrema_estimation_sampling import extrema_estimation_sampling_method


def estimate_scenario_extrema(
    account_drivers_base_forecasts: dts.AccountDriverGroup,
    general_params: GeneralParams,
    all_scenario_forecasts: dict[SPDriverScenarioEnum, dts.DriverGroup],
    extrema_estimation_params: SPExtremaEstimationParams,
    account_forecasting_instance: AbstractAccountForecastingMethod,
    lags: dict[dts.DriverName, int],
) -> dict[SPExtremaCase, dict[dts.DriverName, SPDriverScenarioEnum]]:
    match extrema_estimation_params.extrema_estimation_method:
        case SPExtremaEstimationMethodEnum.AUTO:
            if account_forecasting_instance.is_linear():
                return extrema_estimation_correlation_method(
                    account_drivers_base_forecasts=account_drivers_base_forecasts,
                    general_params=general_params,
                    lags=lags,
                )
            else:
                return extrema_estimation_sampling_method(
                    accounts_drivers_base_forecasts=account_drivers_base_forecasts,
                    general_params=general_params,
                    all_scenario_forecasts=all_scenario_forecasts,
                    estimation_params=extrema_estimation_params.sampling_extrema_estimation_params,
                    account_forecasting_instance=account_forecasting_instance,
                    lags=lags,
                )

        case SPExtremaEstimationMethodEnum.CORRELATION:
            return extrema_estimation_correlation_method(
                account_drivers_base_forecasts=account_drivers_base_forecasts,
                general_params=general_params,
                lags=lags,
            )

        case SPExtremaEstimationMethodEnum.EXACT:
            return extrema_estimation_sampling_method(
                accounts_drivers_base_forecasts=account_drivers_base_forecasts,
                general_params=general_params,
                all_scenario_forecasts=all_scenario_forecasts,
                estimation_params=extrema_estimation_params.exact_extrema_estimation_params,
                account_forecasting_instance=account_forecasting_instance,
                lags=lags,
            )

        case SPExtremaEstimationMethodEnum.SAMPLING:
            return extrema_estimation_sampling_method(
                accounts_drivers_base_forecasts=account_drivers_base_forecasts,
                general_params=general_params,
                all_scenario_forecasts=all_scenario_forecasts,
                estimation_params=extrema_estimation_params.sampling_extrema_estimation_params,
                account_forecasting_instance=account_forecasting_instance,
                lags=lags,
            )

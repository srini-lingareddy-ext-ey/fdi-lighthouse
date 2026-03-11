import datetime

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
)
from lh_v2.forecasting.driver_forecasting import (
    DriverForecastingInput,
    create_driver_forecasts,
)

from .sp_driver_forecasting_methods import SP_DRIVER_PERTURBATION_METHOD_MAP


def make_all_driver_perturbations(
    driver_base_forecast: dts.DriverGroup,
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    lags: dict[dts.DriverName, int],
    sp_df_params: params.SPDriverForecastingParams,
) -> dict[SPDriverScenarioEnum, dts.DriverGroup]:
    cases_to_run = set(SPDriverScenarioEnum) - {SPDriverScenarioEnum.AVG}
    scenarios: dict[
        SPDriverScenarioEnum, dict[dts.DriverName, SPDriverScenarioEnum]
    ] = {
        scenario_case: {
            driver_name: scenario_case
            for driver_name in driver_base_forecast.get_ordered_drivers()
        }
        for scenario_case in cases_to_run
    }

    method_instance = SP_DRIVER_PERTURBATION_METHOD_MAP[
        sp_df_params.selected_perturbation_method
    ](
        base_driver_forecasts=driver_base_forecast,
        training_daterange=training_daterange,
        forecasting_daterange=forecast_daterange,
        params=sp_df_params.get_selected_perturbation_method_params(),
        best_lags=lags,
    )

    perturbed_driver_groups: dict[SPDriverScenarioEnum, dts.DriverGroup] = {}
    for scenario_case, scenario in scenarios.items():
        perturbed_driver_groups[scenario_case] = method_instance.perturb_forecasts(
            scenario
        )
    perturbed_driver_groups[SPDriverScenarioEnum.AVG] = driver_base_forecast

    return perturbed_driver_groups


def create_driver_scenarios(
    accounts_drivers: dts.AccountGroupSelectedDrivers,
    general_params: params.GeneralParams,
    df_params: params.DriverForecastParams,
    sp_df_params: params.SPDriverForecastingParams,
    best_lags: dict[dts.AccountType, dict[dts.DriverName, int]],
) -> dict[dts.AccountType, dict[SPDriverScenarioEnum, dts.DriverGroup]]:
    """
    Create perturbed driver forecasts for the user scenario and all 5 perturbation
    levels per account.

    Returns
    -------
    all_perturbation_levels : dict[AccountType, dict[SPDriverScenarioEnum, DriverGroup]]
        All 5 perturbation levels per account.  Passed downstream so that
        ``create_accounts_scenarios`` can evaluate combos through the real
        trained model to find the true MAX / MIN extrema.
    """
    # Storage for results
    all_perturbation_levels: dict[
        dts.AccountType, dict[SPDriverScenarioEnum, dts.DriverGroup]
    ] = {}

    # Process each account type separately
    for account in accounts_drivers.accounts.get_ordered_accounts():
        # Create base driver forecasts using the training data
        base_forecasts = create_driver_forecasts(
            forecasting_info=DriverForecastingInput(
                drivers=accounts_drivers[account].drivers,
                lags=best_lags[account],
                training_daterange=general_params.get_train_val_daterange(),
                forecast_daterange=general_params.get_testing_daterange(),
            ),
            df_params=df_params,
        )

        all_perturbation_levels[account] = make_all_driver_perturbations(
            driver_base_forecast=base_forecasts,
            training_daterange=general_params.get_train_val_daterange(),
            forecast_daterange=general_params.get_testing_daterange(),
            lags=best_lags[account],
            sp_df_params=sp_df_params,
        )

    return all_perturbation_levels

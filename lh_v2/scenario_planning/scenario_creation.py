import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.datatypes.scenario_planning_types import SPDriverScenarioEnum
from lh_v2.scenario_planning.scenario_planning_types import (
    ScenarioPlanningInput,
    ScenarioPlanningOutput,
)
from lh_v2.scenario_planning.sp_account_forecasting.account_scenario_creation import (
    create_accounts_scenarios,
)
from lh_v2.scenario_planning.sp_driver_forecasting import (
    create_driver_scenarios,
)


def create_scenarios(
    scenario_input: ScenarioPlanningInput,
    lh_params: params.LighthouseParams,
    scenario: dict[dts.DriverName, SPDriverScenarioEnum],
) -> ScenarioPlanningOutput:
    # Create perturbed driver forecasts for the scenario and all 5 perturbation levels per account
    all_perturbation_levels = create_driver_scenarios(
        accounts_drivers=scenario_input.accounts_drivers,
        general_params=lh_params.general_params,
        df_params=lh_params.driver_forecast_params,
        sp_df_params=lh_params.scenario_planning_params.sp_driver_forecasting_params,
        best_lags=scenario_input.lags,
    )

    # Train models, find extrema through real models, produce forecasts
    perturbed_accounts, extrema_accounts = create_accounts_scenarios(
        accounts_drivers_base_forecasts=scenario_input.accounts_drivers,
        all_perturbation_levels=all_perturbation_levels,
        general_params=lh_params.general_params,
        sp_extrema_estimation_params=lh_params.scenario_planning_params.sp_extrema_estimation_params,
        scenario=scenario,
        lags=scenario_input.lags,
        selected_models=scenario_input.selected_model,
        best_params=scenario_input.best_params,
    )

    return ScenarioPlanningOutput(
        perturbed_accounts=perturbed_accounts,
        extrema_accounts=extrema_accounts,
    )

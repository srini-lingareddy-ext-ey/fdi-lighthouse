import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.data_analysis.scenario_planning.account_perturbation_util import (
    plot_all_scenarios,
)
from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    AccountForecastingMethodEnum,
)
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverForecastingMethodEnum,
)
from lh_v2.scenario_planning.sp_account_forecasting.account_scenario_creation import (
    create_account_scenarios_util,
)
from lh_v2.scenario_planning.sp_account_forecasting.scenario_extrema_estimation import (
    estimate_scenario_extrema,
)
from lh_v2.scenario_planning.sp_driver_forecasting import (
    make_all_driver_perturbations,
)


def plot_account_perturbation_estimation_info(
    account_drivers: dts.AccountDriverGroup,
    general_params: params.GeneralParams,
    af_params: params.AccountForecastParams,
    sp_params: params.ScenarioPlanningParams,
    account_forecasting_method: AccountForecastingMethodEnum,
    perturbation_method: SPDriverForecastingMethodEnum,
    lags: dict[dts.DriverName, int] | None = None,
    b_plot_historicals: bool = False,
):
    if lags is None:
        lags = {driver: 0 for driver in account_drivers.drivers.get_ordered_drivers()}

    sp_params.sp_driver_forecasting_params.selected_perturbation_method = (
        perturbation_method
    )

    perturbed_drivers = make_all_driver_perturbations(
        driver_base_forecast=account_drivers.drivers,
        training_daterange=general_params.get_train_val_daterange(),
        forecast_daterange=general_params.get_testing_daterange(),
        lags=lags,
        sp_df_params=sp_params.sp_driver_forecasting_params,
    )

    lst_scenarios, arr_scenarios, account_forecasting_instance = (
        create_account_scenarios_util(
            account_drivers_base_forecasts=account_drivers,
            all_scenario_forecasts=perturbed_drivers,
            general_params=general_params,
            lags=lags,
            selected_model=account_forecasting_method,
            best_params=af_params,
        )
    )

    extrema_cases = estimate_scenario_extrema(
        account_drivers_base_forecasts=account_drivers,
        general_params=general_params,
        all_scenario_forecasts=perturbed_drivers,
        extrema_estimation_params=sp_params.sp_extrema_estimation_params,
        account_forecasting_instance=account_forecasting_instance,
        lags=lags,
    )

    plot_all_scenarios(
        base_forecast=account_drivers.account,
        forecast_daterange=general_params.get_testing_daterange(),
        arr_scenarios=arr_scenarios,
        extrema_cases=extrema_cases,
        lst_scenarios=lst_scenarios,
        b_plot_historicals=b_plot_historicals,
    )

    return

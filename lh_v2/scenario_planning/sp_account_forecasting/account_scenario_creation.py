from itertools import product

import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.forecasting_types.account_forecasting_types import (
    AccountForecastingMethodEnum,
)
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
    SPExtremaCase,
)
from lh_v2.forecasting.account_forecasting.account_forecasting_methods import (
    ACCOUNT_FORECASTING_METHOD_MAP,
    AbstractAccountForecastingMethod,
)
from lh_v2.params import AccountForecastParams, GeneralParams, SPExtremaEstimationParams
from lh_v2.scenario_planning.sp_account_forecasting.scenario_extrema_estimation import (
    estimate_scenario_extrema,
)
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif

from .scenario_handling_util import scenario_to_driver_group


def create_account_scenarios(
    account_drivers_base_forecasts: dts.AccountDriverGroup,
    all_scenario_forecasts: dict[SPDriverScenarioEnum, dts.DriverGroup],
    scenario: dict[dts.DriverName, SPDriverScenarioEnum],
    general_params: GeneralParams,
    sp_extrema_estimation_params: SPExtremaEstimationParams,
    lags: dict[dts.DriverName, int],
    selected_model: AccountForecastingMethodEnum,
    best_params: AccountForecastParams,
) -> tuple[dts.AccountInfo, dict[SPExtremaCase, dts.AccountInfo]]:
    # Initialize and train the account forecasting model
    forecasting_method_instance = ACCOUNT_FORECASTING_METHOD_MAP[selected_model](
        info=account_drivers_base_forecasts,
        best_lags=lags,
        training_daterange=general_params.get_train_val_daterange(),
        forecast_daterange=general_params.get_testing_daterange(),
        model_params=best_params[selected_model],
    )
    forecasting_method_instance.train()

    # Compute AVG correction
    forecasting_method_instance.set_forecasting_input_data(
        new_drivers_data=all_scenario_forecasts[SPDriverScenarioEnum.AVG]
    )
    avg_scenario = forecasting_method_instance.forecast()
    correction = account_drivers_base_forecasts.account.arr - avg_scenario.arr

    # Forecast account using actual given scenario
    forecasting_method_instance.set_forecasting_input_data(
        new_drivers_data=scenario_to_driver_group(
            scenario=scenario,
            all_scenario_forecasts=all_scenario_forecasts,
        )
    )
    actual_scenario_acc = forecasting_method_instance.forecast()

    actual_scenario_acc.arr += correction

    # Estimate extrema scenarios
    extrema_scenarios = estimate_scenario_extrema(
        account_drivers_base_forecasts=account_drivers_base_forecasts,
        general_params=general_params,
        all_scenario_forecasts=all_scenario_forecasts,
        extrema_estimation_params=sp_extrema_estimation_params,
        account_forecasting_instance=forecasting_method_instance,
        lags=lags,
    )

    extrema_forecasts: dict[SPExtremaCase, dts.AccountInfo] = {}
    for case, scenario_dict in extrema_scenarios.items():
        forecasting_method_instance.set_forecasting_input_data(
            new_drivers_data=scenario_to_driver_group(
                scenario=scenario_dict,
                all_scenario_forecasts=all_scenario_forecasts,
            )
        )
        extrema_forecasts[case] = forecasting_method_instance.forecast()
        extrema_forecasts[case].arr += correction

    return actual_scenario_acc, extrema_forecasts


def create_accounts_scenarios(
    accounts_drivers_base_forecasts: dts.AccountGroupSelectedDrivers,
    all_perturbation_levels: dict[
        dts.AccountType, dict[SPDriverScenarioEnum, dts.DriverGroup]
    ],
    general_params: GeneralParams,
    sp_extrema_estimation_params: SPExtremaEstimationParams,
    scenario: dict[dts.DriverName, SPDriverScenarioEnum],
    lags: dict[dts.AccountType, dict[dts.DriverName, int]],
    selected_models: dict[dts.AccountType, AccountForecastingMethodEnum],
    best_params: dict[dts.AccountType, AccountForecastParams],
) -> tuple[
    dts.AccountGroupInfo,
    dict[SPExtremaCase, dts.AccountGroupInfo],
]:
    """

    Returns
    -------
    scenario_forecasts_group : AccountGroupInfo
        Account forecasts for the user-defined scenario.
    extrema_forecasts_groups : dict[SPExtremaCase, AccountGroupInfo]
        Account forecasts for the MAX / MIN extrema.
    extrema_drivers_groups : dict[SPExtremaCase, ClassifiedDriverGroups]
        The driver groups corresponding to the chosen extrema combos.
    """
    lst_scenario_forecasts: list[dts.AccountInfo] = []
    dict_extrema_forecasts: dict[SPExtremaCase, list[dts.AccountInfo]] = {
        case: [] for case in SPExtremaCase
    }

    for account in accounts_drivers_base_forecasts.accounts.get_ordered_accounts():
        (
            scenario_forecast,
            extrema_forecasts,
        ) = create_account_scenarios(
            account_drivers_base_forecasts=accounts_drivers_base_forecasts[account],
            all_scenario_forecasts=all_perturbation_levels[account],
            scenario=scenario,
            general_params=general_params,
            sp_extrema_estimation_params=sp_extrema_estimation_params,
            lags=lags[account],
            selected_model=selected_models[account],
            best_params=best_params[account],
        )
        lst_scenario_forecasts.append(scenario_forecast)
        for case in SPExtremaCase:
            dict_extrema_forecasts[case].append(extrema_forecasts[case])

    scenario_forecasts_group = dts.AccountGroupInfo.from_account_lst(
        account_lst=lst_scenario_forecasts,
    )
    dict_extrema_forecasts_group = {
        case: dts.AccountGroupInfo.from_account_lst(account_lst=extrema_lst)
        for case, extrema_lst in dict_extrema_forecasts.items()
    }

    return scenario_forecasts_group, dict_extrema_forecasts_group


def create_account_scenarios_util(
    account_drivers_base_forecasts: dts.AccountDriverGroup,
    all_scenario_forecasts: dict[SPDriverScenarioEnum, dts.DriverGroup],
    general_params: GeneralParams,
    lags: dict[dts.DriverName, int],
    selected_model: AccountForecastingMethodEnum,
    best_params: AccountForecastParams,
) -> tuple[
    list[dict[dts.DriverName, SPDriverScenarioEnum]],
    ArrayF,
    AbstractAccountForecastingMethod,
]:
    forecasting_method_instance = ACCOUNT_FORECASTING_METHOD_MAP[selected_model](
        info=account_drivers_base_forecasts,
        best_lags=lags,
        training_daterange=general_params.get_train_val_daterange(),
        forecast_daterange=general_params.get_testing_daterange(),
        model_params=best_params[selected_model],
    )

    forecasting_method_instance.train()

    all_scenario_combos = tuple(
        product(
            *[
                [case for case in SPDriverScenarioEnum]
                for _ in range(
                    len(account_drivers_base_forecasts.drivers.get_ordered_drivers())
                )
            ]
        )
    )

    n_forecast_vals = (
        month_dif(
            start_date=general_params.testing_start_date,
            end_date=general_params.testing_end_date,
        )
        + 1
    )

    lst_scenario_out: list[dict[dts.DriverName, SPDriverScenarioEnum]] = []

    arr_input: ArrayF = np.zeros(
        (
            len(all_scenario_combos),
            len(account_drivers_base_forecasts.drivers),
            n_forecast_vals,
        ),
        dtype=account_drivers_base_forecasts.np_dtype,
    )

    for idx1, scenario_tpl in enumerate(all_scenario_combos):
        for idx2 in range(len(account_drivers_base_forecasts.drivers)):
            arr_input[idx1, idx2, :] = all_scenario_forecasts[scenario_tpl[idx2]].arr[
                idx2, -n_forecast_vals:
            ]

    arr_output = forecasting_method_instance.apply_vectorized(arr_input=arr_input)

    for idx1, scenario_tpl in enumerate(all_scenario_combos):
        lst_scenario_out.append(
            {
                driver: scenario_tpl[idx2]
                for idx2, driver in enumerate(
                    account_drivers_base_forecasts.drivers.get_ordered_drivers()
                )
            }
        )

    avg_idx = lst_scenario_out.index(
        {
            driver: SPDriverScenarioEnum.AVG
            for driver in account_drivers_base_forecasts.drivers.get_ordered_drivers()
        }
    )
    diff = (
        account_drivers_base_forecasts.account.arr[-n_forecast_vals:]
        - arr_output[avg_idx, :]
    )

    arr_output += diff.reshape(1, -1)

    return lst_scenario_out, arr_output, forecasting_method_instance

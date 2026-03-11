import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import SPDriverScenarioEnum
from lh_v2.shared import ArrayF, ArrayI


def combo_to_scenario_dict(
    combo: ArrayI, drivers_ordered: list[dts.DriverName]
) -> dict[dts.DriverName, SPDriverScenarioEnum]:
    scenario_enums_ordered = list(SPDriverScenarioEnum)
    return {
        driver: scenario_enums_ordered[level]
        for driver, level in zip(drivers_ordered, combo)
    }


def combos_to_scenario_arr(
    combos: ArrayI,
    all_scenarios_forecast: dict[SPDriverScenarioEnum, dts.DriverGroup],
) -> ArrayF:
    """
    Convert an array of scenario level indices into an array of driver
    forecasts by looking up the corresponding forecast for each driver
    and scenario level.

    Parameters
    ----------
    combos : ArrayI
        Array of shape (n_combos, n_drivers) containing scenario level
        indices for each driver in each combo.
    all_scenarios_forecast : dict[SPDriverScenarioEnum, DriverGroup]
        Dictionary mapping each scenario enum to a DriverGroup containing
        the forecasts for all drivers under that scenario.
        These driver groups are required to contain only the forecasts,
        no previous historical data, since we will be using the arr directly.
        See apply_daterange() method on dts.DriverGroup.

    Returns
    -------
    ArrayF
        Array of shape (n_combos, n_drivers, n_time) containing the driver
        forecasts for each combo.
    """
    scenario_enums_ordered = list(SPDriverScenarioEnum)
    n_drivers = all_scenarios_forecast[scenario_enums_ordered[0]].arr.shape[0]
    n_vals = all_scenarios_forecast[scenario_enums_ordered[0]].arr.shape[1]
    arr_all_scenarios = np.zeros(
        (len(scenario_enums_ordered), n_drivers, n_vals),
        dtype=all_scenarios_forecast[scenario_enums_ordered[0]].np_dtype,
    )

    for idx, scenario_enum in enumerate(scenario_enums_ordered):
        arr_all_scenarios[idx] = all_scenarios_forecast[scenario_enum].arr

    arr_forecast_combos = np.zeros(
        (combos.shape[0], n_drivers, n_vals), dtype=arr_all_scenarios.dtype
    )
    arr_drivers_inds: ArrayI = np.arange(n_drivers)
    for combo_idx, combo in enumerate(combos):
        arr_forecast_combos[combo_idx] = arr_all_scenarios[combo, arr_drivers_inds]

    return arr_forecast_combos

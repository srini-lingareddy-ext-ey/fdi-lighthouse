import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import SPDriverScenarioEnum


def scenario_to_driver_group(
    scenario: dict[dts.DriverName, SPDriverScenarioEnum],
    all_scenario_forecasts: dict[SPDriverScenarioEnum, dts.DriverGroup],
) -> dts.DriverGroup:
    """
    Convert a scenario dict mapping driver to scenario enum into a DriverGroup
    containing the corresponding forecasts for each driver.

    Parameters
    ----------
    scenario : dict[dts.DriverName, SPDriverScenarioEnum]
        Dictionary mapping each driver to a scenario enum.
    all_scenario_forecasts : dict[SPDriverScenarioEnum, DriverGroup]
        Dictionary mapping each scenario enum to a DriverGroup containing
        the forecasts for all drivers under that scenario.

    Returns
    -------
    DriverGroup
        DriverGroup containing the forecasts for each driver under the given scenario.
    """
    drivers_ordered = all_scenario_forecasts[
        SPDriverScenarioEnum.AVG
    ].get_ordered_drivers()

    lst_drivers: list[dts.Driver] = []
    for driver in drivers_ordered:
        lst_drivers.append(
            all_scenario_forecasts[
                scenario.get(driver, SPDriverScenarioEnum.AVG)
            ].get_driver(driver_name=driver)
        )

    return dts.DriverGroup.from_driver_lst(
        driver_lst=lst_drivers,
        np_dtype=all_scenario_forecasts[SPDriverScenarioEnum.AVG].np_dtype,
    )

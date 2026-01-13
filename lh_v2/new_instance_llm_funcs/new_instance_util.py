import pathlib as pth
from typing import Any, Optional

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.driver_analysis import DriverAnalysisInput, analyze_drivers_full
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import LighthouseParams, parse_yaml


def initialize(
    config_info: pth.Path | dict[str, Any],
    account_data_pth: pth.Path,
    driver_data_pth: pth.Path,
) -> tuple[LighthouseParams, DriverAnalysisInput]:
    # Read configuration parameters
    if isinstance(config_info, pth.Path):
        params = parse_yaml(config_info)
    else:
        params = LighthouseParams(**config_info)

    # Load account and driver data
    dr_data = load_data_driver_ranking(
        lh_params=params, account_source=account_data_pth, driver_source=driver_data_pth
    )

    return params, dr_data


def handle_driver_analysis(
    accounts_drivers_info: DriverAnalysisInput,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
    selected_drivers: Optional[
        dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]]
    ] = None,
    drivers_lags: Optional[dict[dts.AccountType, dict[dts.DriverName, int]]] = None,
) -> tuple[
    dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]],
    dict[dts.AccountType, dict[dts.DriverName, int]],
    dict[dts.AccountType, dict[dts.DriverName, dts.DriverClassification]],
]:
    if selected_drivers is None or drivers_lags is None:
        # Perform full driver analysis to select drivers and optimal lags
        analysis_results = analyze_drivers_full(
            accounts_drivers_info=accounts_drivers_info,
            general_params=general_params,
            da_params=da_params,
        )
        selected_drivers = analysis_results.selected_drivers
        drivers_lags = analysis_results.format_lags()
        drivers_classifications = analysis_results.format_classifications()

    else:
        drivers_classifications: dict[
            dts.AccountType, dict[dts.DriverName, dts.DriverClassification]
        ] = {}
        for account in selected_drivers.keys():
            drivers_classifications[account] = {}
            for class_ in selected_drivers[account].keys():
                for driver in selected_drivers[account][class_]:
                    drivers_classifications[account][driver] = class_

    return selected_drivers, drivers_lags, drivers_classifications

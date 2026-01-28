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
    """
    Initialize Lighthouse parameters and load driver analysis input data.

    Parameters
    ----------
    config_info : pathlib.Path or dict[str, Any]
        Path to YAML configuration file or dictionary of configuration parameters.
    account_data_pth : pathlib.Path
        Path to the account data source file.
    driver_data_pth : pathlib.Path
        Path to the driver data source file.

    Returns
    -------
    params : LighthouseParams
        Parsed Lighthouse configuration parameters.
    dr_data : DriverAnalysisInput
        Loaded driver ranking data including account and driver information.
    """
    # Read configuration parameters
    # Check if config_info is a file path or a dictionary
    if isinstance(config_info, pth.Path):
        # Parse YAML file into LighthouseParams object
        params = parse_yaml(config_info)
    else:
        # Convert dictionary directly into LighthouseParams object
        params = LighthouseParams(**config_info)

    # Load account and driver data
    # Use the parsed parameters to load and structure the driver ranking data
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
    """
    Handle driver analysis by either using provided selections or performing full analysis.

    If driver selections and lags are not provided, performs a complete driver analysis
    to determine optimal drivers and their lag values. Otherwise, uses the provided
    selections and derives driver classifications from them.

    Parameters
    ----------
    accounts_drivers_info : DriverAnalysisInput
        Input data containing account and driver information for analysis.
    general_params : params.GeneralParams
        General configuration parameters for the analysis.
    da_params : params.DriverAnalysisParams
        Specific parameters for driver analysis configuration.
    selected_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]], optional
        Pre-selected drivers organized by account type and classification.
        If None, driver analysis will be performed to select drivers. Default is None.
    drivers_lags : dict[AccountType, dict[DriverName, int]], optional
        Lag values for each driver by account type.
        If None, optimal lags will be determined through analysis. Default is None.

    Returns
    -------
    selected_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]]
        Selected drivers organized by account type and classification.
    drivers_lags : dict[AccountType, dict[DriverName, int]]
        Optimal lag values for each driver by account type.
    drivers_classifications : dict[AccountType, dict[DriverName, DriverClassification]]
        Classification mapping for each driver by account type.
    """
    # Check if we need to perform driver analysis or use provided selections
    if selected_drivers is None or drivers_lags is None:
        # Perform full driver analysis to select drivers and optimal lags
        # This runs statistical analysis to determine the best drivers for each account
        analysis_results = analyze_drivers_full(
            accounts_drivers_info=accounts_drivers_info,
            general_params=general_params,
            da_params=da_params,
        )
        # Extract the selected drivers from analysis results
        selected_drivers = analysis_results.selected_drivers
        # Format and extract optimal lag values for each driver
        drivers_lags = analysis_results.format_lags()
        # Format and extract classification for each driver
        drivers_classifications = analysis_results.format_classifications()

    else:
        # Use pre-selected drivers and lags, but need to derive classifications
        # Initialize empty dictionary to store driver classifications by account
        drivers_classifications: dict[
            dts.AccountType, dict[dts.DriverName, dts.DriverClassification]
        ] = {}
        # Iterate through each account type in the selected drivers
        for account in selected_drivers.keys():
            # Initialize dictionary for this account's driver classifications
            drivers_classifications[account] = {}
            # Iterate through each classification category (e.g., leading, lagging)
            for class_ in selected_drivers[account].keys():
                # Iterate through each driver in this classification
                for driver in selected_drivers[account][class_]:
                    # Map the driver to its classification for easy lookup
                    drivers_classifications[account][driver] = class_

    return selected_drivers, drivers_lags, drivers_classifications

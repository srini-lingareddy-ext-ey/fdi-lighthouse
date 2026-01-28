import pathlib as pth
from typing import Any

from lh_v2.driver_analysis import DriverAnalysisOutputLLM, analyze_drivers_llm

from .new_instance_util import initialize


def select_drivers_llm_new(
    config_info: pth.Path | dict[str, Any],
    account_data_pth: pth.Path,
    driver_data_pth: pth.Path,
) -> DriverAnalysisOutputLLM:
    """
    Select drivers using LLM-based analysis for new Lighthouse instances.

    This function initializes Lighthouse parameters, loads account and driver data,
    and performs driver analysis specifically formatted for LLM-based driver selection.
    The output includes driver rankings and statistics suitable for LLM interpretation.

    Parameters
    ----------
    config_info : pathlib.Path or dict[str, Any]
        Path to YAML configuration file or dictionary of configuration parameters
        containing general and driver analysis parameters.
    account_data_pth : pathlib.Path
        Path to the account data source file.
    driver_data_pth : pathlib.Path
        Path to the driver data source file.

    Returns
    -------
    analysis_results : DriverAnalysisOutputLLM
        Driver analysis results formatted for LLM processing, including driver
        rankings, correlations, and statistical metrics for each account type.
    """
    # Initialize parameters and load data
    # Parse configuration and load account/driver data into structured format
    lh_params, dr_data = initialize(
        config_info=config_info,
        account_data_pth=account_data_pth,
        driver_data_pth=driver_data_pth,
    )

    # Perform initial driver analysis to be sent to LLM for driver selection
    # Run analysis that generates rankings and statistics in LLM-friendly format
    # This includes correlation metrics, importance scores, and lag information
    analysis_results = analyze_drivers_llm(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
    )

    return analysis_results

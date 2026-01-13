import pathlib as pth
from typing import Any

from lh_v2.driver_analysis import DriverAnalysisOutputLLM, analyze_drivers_llm

from .new_instance_util import initialize


def select_drivers_llm_new(
    config_info: pth.Path | dict[str, Any],
    account_data_pth: pth.Path,
    driver_data_pth: pth.Path,
) -> DriverAnalysisOutputLLM:
    # Initialize parameters and load data
    lh_params, dr_data = initialize(
        config_info=config_info,
        account_data_pth=account_data_pth,
        driver_data_pth=driver_data_pth,
    )

    # Perform initial driver analysis to be sent to LLM for driver selection
    analysis_results = analyze_drivers_llm(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
    )

    return analysis_results

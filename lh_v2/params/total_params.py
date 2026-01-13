import logging
import pathlib as pth
from typing import Any

import yaml
from pydantic import Field

import lh_v2.datatypes as dts
from lh_v2.util import BaseParamsModel

from .driver_analysis_params import DriverAnalysisParams
from .forecasting_params import AccountForecastParams, DriverForecastParams
from .general_params import (
    GeneralParams,
)
from .io_params import DataLoadingParams
from .logging_params import LoggingParams, setup_logging

logger = logging.getLogger(__name__)


class LighthouseParams(BaseParamsModel):
    """
    Main configuration class for Lighthouse forecasting system.

    This class aggregates all parameter configurations required for the Lighthouse
    forecasting pipeline, including data loading, general settings, driver analysis,
    and forecasting parameters.

    Attributes
    ----------
    accounts : list[dts.AccountType]
        List of account types to process. Defaults to a single 'volume' account type.
    segment : dts.SegmentType
        Market segment for analysis. Defaults to 'Residential'.
    region : dts.RegionType
        Geographic region for analysis. Defaults to 'North America'.
    load_data_params : DataLoadingParams
        Configuration parameters for data loading operations.
    general_params : GeneralParams
        General configuration parameters for the forecasting system.
    driver_analysis_params : DriverAnalysisParams
        Configuration parameters for driver analysis phase.
    driver_forecast_params : DriverForecastParams
        Configuration parameters for driver forecasting phase.
    account_forecast_params : AccountForecastParams
        Configuration parameters for account-level forecasting phase.

    Examples
    --------
    >>> params = LighthouseParams()
    >>> params.segment = dts.SegmentType('Commercial')
    >>> params.accounts = [dts.AccountType('revenue'), dts.AccountType('volume')]
    """

    accounts: list[dts.AccountType] = Field(
        default_factory=lambda: [dts.AccountType('volume')]
    )
    segment: dts.ProductType = Field(
        default_factory=lambda: dts.ProductType('Residential')
    )
    region: dts.LocationType = Field(
        default_factory=lambda: dts.LocationType('North America')
    )
    load_data_params: DataLoadingParams = Field(default_factory=DataLoadingParams)
    general_params: GeneralParams = Field(default_factory=GeneralParams)
    driver_analysis_params: DriverAnalysisParams = Field(
        default_factory=DriverAnalysisParams
    )
    driver_forecast_params: DriverForecastParams = Field(
        default_factory=DriverForecastParams
    )
    account_forecast_params: AccountForecastParams = Field(
        default_factory=AccountForecastParams
    )


def _parse_primary_params(yaml_dict: dict[str, Any]) -> LighthouseParams:
    """
    Helper function to parse primary parameters from a YAML dictionary.
    """
    default_params = LighthouseParams()

    accounts: list[dts.AccountType] = []
    for acc in yaml_dict.get('accounts', default_params.accounts):
        accounts.append(dts.AccountType(acc))

    segment = dts.ProductType(yaml_dict.get('segment', default_params.segment))
    region = dts.LocationType(yaml_dict.get('region', default_params.region))
    general_params = GeneralParams(**yaml_dict.get('general_params', {}))

    default_params.accounts = accounts
    default_params.segment = segment
    default_params.region = region
    default_params.general_params = general_params
    return default_params


def parse_yaml(yaml_info: pth.Path | dict[str, Any]) -> LighthouseParams:
    """
    Parse a YAML file and return a ScenarioPlanningParams object.

    Parameters
    ----------
    yaml_path : pathlib.Path
        The path to the YAML file to be parsed.

    Returns
    -------
    ScenarioPlanningParams
        An instance of ScenarioPlanningParams populated with the data from the YAML file.

    Notes
    -----
    - If the YAML file contains the key "use_default" set to True, a default ScenarioPlanningParams
      object is returned.
    - The function expects the YAML file to contain specific keys such as "accounts", "segments",
      "regions", "general", "testing", "lagging", "driverRanking", "Collinearity", "driverForecasts",
      "accountForecasts", and "output". Each of these keys is used to populate the corresponding
      parameters in the ScenarioPlanningParams object.
    - The function relies on helper functions like `parse_general_params`, `parse_testing_params`,
      `parse_lag_params`, `parse_ranking_params`, `parse_collinearity_params`,
      `parse_driver_forecast_params`, `parse_account_forecast_params`, and `parse_output_params`
      to process specific sections of the YAML file.
    """
    # Load the YAML file into a dictionary

    yaml_dict: dict[str, Any]
    if isinstance(yaml_info, dict):
        yaml_dict = yaml_info
    else:
        with open(yaml_info, 'r') as stream:
            yaml_dict = yaml.safe_load(stream)

    # Check if the configuration specifies using default parameters
    if yaml_dict.get('use_default', False):
        # Initialize logging with default settings
        setup_logging(LoggingParams())
        # Return a LighthouseParams object with all default values
        return LighthouseParams()

    if yaml_dict.get('use_default_params', False):
        # Initialize logging with default settings
        setup_logging(LoggingParams())
        # Parse primary parameters and return with defaults for the rest
        return _parse_primary_params(yaml_dict)

    # Initialize logging with custom settings from the YAML file (if provided)
    setup_logging(LoggingParams(**(yaml_dict.get('logging', {}))))
    # Create and return a LighthouseParams object populated with values from the YAML file
    return LighthouseParams(**yaml_dict)

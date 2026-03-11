"""
Data loading module for Lighthouse forecasting system.

This module provides functions to load account and driver data from various sources
using configurable data loading methods. It supports different data loading backends
and formats the data into structures suitable for driver analysis and forecasting.
"""

import time
from typing import Any

import numpy as np

import lh_v2.datatypes.data_loader_types as ldt
import lh_v2.params as params
from lh_v2.datatypes import (
    AccountGroupClassifiedDriverGroups,
    ClassifiedDriverGroups,
    DriverClassification,
    DriverGroup,
)
from lh_v2.driver_analysis import DriverAnalysisInput
from lh_v2.util import get_logger

from .data_loading_methods import (
    AbstractDataLoader,
    PolarsDataLoader,
)

logger = get_logger(__name__)

DATA_LOADING_METHOD_MAP: dict[ldt.DataLoadingMethodEnum, type[AbstractDataLoader]] = {
    ldt.DataLoadingMethodEnum.POLARS: PolarsDataLoader,
}


def load_data_general(
    lh_params: params.LighthouseParams, account_source: Any, driver_source: Any
) -> AccountGroupClassifiedDriverGroups:
    """
    Load account and driver data for driver ranking and analysis.

    This function orchestrates the data loading process by selecting the appropriate
    data loading method based on configuration, loading both account and driver data,
    and structuring them into a DriverAnalysisInput object suitable for driver
    analysis operations.

    Parameters
    ----------
    lh_params : params.LighthouseParams
        Lighthouse parameters containing configuration for data loading, including
        the selected data loading method and other specifications.
    account_source : Any
        Source for account data. Type depends on the selected data loading method.
        Can be a file path, DataFrame, or other data source supported by the loader.
    driver_source : Any
        Source for driver data. Type depends on the selected data loading method.
        Can be a file path, DataFrame, or other data source supported by the loader.

    Returns
    -------
    DriverAnalysisInput
        Structured input data for driver analysis containing:
        - accounts: Loaded and processed account information
        - classified_drivers: Loaded and processed driver data organized by classification
        - np_dtype: NumPy data type used for numerical arrays (float32)

    Notes
    -----
    The function:
    1. Selects the appropriate data loader based on lh_params configuration
    2. Loads account data and logs timing information
    3. Loads driver data and logs timing information
    4. Combines both into a DriverAnalysisInput structure
    5. Logs total data loading time

    The data loading method is determined by `lh_params.load_data_params.selected_method`,
    which maps to a specific implementation in DATA_LOADING_METHOD_MAP.

    See Also
    --------
    DriverAnalysisInput : Output data structure containing loaded data.
    AbstractDataLoader : Base class for data loading implementations.
    PolarsDataLoader : Polars-based data loading implementation.

    Examples
    --------
    >>> lh_params = params.LighthouseParams(...)
    >>> data = load_data_driver_ranking(
    ...     lh_params=lh_params,
    ...     account_source='data/accounts.parquet',
    ...     driver_source='data/drivers.csv'
    ... )
    >>> data.accounts.arr.shape
    (5, 36)  # 5 accounts, 36 time periods
    >>> data.classified_drivers.arr.shape
    (50, 36)  # 50 total drivers across all classifications, 36 time periods
    """
    # Instantiate the data loader based on the selected method from parameters
    data_loading_method: AbstractDataLoader = DATA_LOADING_METHOD_MAP[
        lh_params.load_data_params.selected_method
    ](lh_params=lh_params)

    # Log which data loading method is being used
    logger.info(
        f'Loading data using method: {lh_params.load_data_params.selected_method.value}'
    )

    # Track overall data loading time and individual step times
    start_time = time.time()
    last_time = time.time()

    # Load account data from the specified source
    accounts_info = data_loading_method.load_account_data(source=account_source)
    # Log time taken to load account data
    logger.timing(f'Account data loaded in {time.time() - last_time:.4f} seconds.')

    # Reset timer for next step
    last_time = time.time()

    # Load driver data from the specified source
    drivers_info = data_loading_method.load_driver_data(source=driver_source)
    lst_driver_groups: list[DriverGroup] = []
    dict_class_map: dict[DriverClassification, int] = {}

    for idx, driver_classification in enumerate(
        sorted(drivers_info.get_ordered_classifications())
    ):
        lst_driver_groups.append(
            drivers_info[driver_classification].order(
                driver_ordering=sorted(
                    drivers_info.get_ordered_drivers(
                        classification=driver_classification
                    )
                )
            )
        )
        dict_class_map[driver_classification] = idx

    # Log time taken to load driver data
    logger.timing(f'Driver data loaded in {time.time() - last_time:.4f} seconds.')

    # Log total time for complete data loading process
    logger.timing(f'Total data loading time: {time.time() - start_time:.4f} seconds.')

    # Return structured data ready for driver analysis
    return AccountGroupClassifiedDriverGroups(
        accounts=accounts_info,
        classified_drivers=ClassifiedDriverGroups.from_driver_group_lst(
            driver_group_lst=lst_driver_groups, classification_groups=dict_class_map
        ),
        np_dtype=np.float32,
    )


def load_data_driver_ranking(
    lh_params: params.LighthouseParams, account_source: Any, driver_source: Any
) -> DriverAnalysisInput:
    """
    Load account and driver data for driver ranking and analysis.

    This function orchestrates the data loading process by selecting the appropriate
    data loading method based on configuration, loading both account and driver data,
    and structuring them into a DriverAnalysisInput object suitable for driver
    analysis operations.

    Parameters
    ----------
    lh_params : params.LighthouseParams
        Lighthouse parameters containing configuration for data loading, including
        the selected data loading method and other specifications.
    account_source : Any
        Source for account data. Type depends on the selected data loading method.
        Can be a file path, DataFrame, or other data source supported by the loader.
    driver_source : Any
        Source for driver data. Type depends on the selected data loading method.
        Can be a file path, DataFrame, or other data source supported by the loader.

    Returns
    -------
    DriverAnalysisInput
        Structured input data for driver analysis containing:
        - accounts: Loaded and processed account information
        - classified_drivers: Loaded and processed driver data organized by classification
        - np_dtype: NumPy data type used for numerical arrays (float32)

    Notes
    -----
    The function:
    1. Selects the appropriate data loader based on lh_params configuration
    2. Loads account data and logs timing information
    3. Loads driver data and logs timing information
    4. Combines both into a DriverAnalysisInput structure
    5. Logs total data loading time

    The data loading method is determined by `lh_params.load_data_params.selected_method`,
    which maps to a specific implementation in DATA_LOADING_METHOD_MAP.

    See Also
    --------
    DriverAnalysisInput : Output data structure containing loaded data.
    AbstractDataLoader : Base class for data loading implementations.
    PolarsDataLoader : Polars-based data loading implementation.

    Examples
    --------
    >>> lh_params = params.LighthouseParams(...)
    >>> data = load_data_driver_ranking(
    ...     lh_params=lh_params,
    ...     account_source='data/accounts.parquet',
    ...     driver_source='data/drivers.csv'
    ... )
    >>> data.accounts.arr.shape
    (5, 36)  # 5 accounts, 36 time periods
    >>> data.classified_drivers.arr.shape
    (50, 36)  # 50 total drivers across all classifications, 36 time periods
    """
    # Instantiate the data loader based on the selected method from parameters
    acc_driv_info = load_data_general(
        lh_params=lh_params, account_source=account_source, driver_source=driver_source
    )

    # Return structured data ready for driver analysis
    return DriverAnalysisInput(
        accounts=acc_driv_info.accounts,
        classified_drivers=acc_driv_info.classified_drivers,
        np_dtype=acc_driv_info.np_dtype,
    )

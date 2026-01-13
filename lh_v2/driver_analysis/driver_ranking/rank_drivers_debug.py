import time
from typing import Sequence

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params as params
from lh_v2.shared import ArrayF, ArrayI
from lh_v2.util import get_logger

from .driver_ranking_methods import (
    AbstractRankingMethod,
)
from .driver_ranking_util import (
    order_drivers_allow_ties,
    order_drivers_no_ties,
    rank_array_to_ranking_dict,
)
from .rank_drivers import RANKING_METHOD_MAP, run_methods, select_methods

logger = get_logger(__name__)


def _format_results_int(
    drivers_results: ArrayI,
    ordered_drivers: Sequence[dts.DriverName],
    selected_methods: Sequence[rdt.DriverRankingEnum],
) -> dict[str, dict[dts.DriverName, int]]:
    """
    Format driver ranking results into a structured dictionary with integer rankings.

    This function organizes the integer ranking results from multiple methods
    into a nested dictionary structure for easier access and analysis. It also
    computes an average ranking across all methods.

    Parameters
    ----------
    drivers_results : ArrayI
        2D array of integer rankings where rows correspond to drivers and columns
        correspond to different ranking methods.
    ordered_drivers : Sequence[dts.DriverName]
        Sequence of driver names in the same order as the rows in drivers_results.
    selected_methods : Sequence[rdt.DriverRankingEnum]
        Ranking methods that were used, corresponding to columns in drivers_results.

    Returns
    -------
    dict[str, dict[dts.DriverName, int]]
        Formatted ranking results organized by method name and driver name.
        Includes individual method rankings and an 'AVG' key containing the
        mean ranking across all methods (converted to integers).
    """
    # Get the string names for each ranking method from the method map
    method_names: list[str] = [
        RANKING_METHOD_MAP[method].name() for method in selected_methods
    ]

    # Initialize dictionary to store rankings for each method and driver
    formatted_results: dict[str, dict[dts.DriverName, int]] = {}

    # Iterate through each ranking method
    for idx1, method in enumerate(method_names):
        formatted_results[method] = {}
        # For each driver, store their ranking from this method
        for idx2, driver in enumerate(ordered_drivers):
            # Extract the ranking from the results array: drivers_results[driver_idx, method_idx]
            formatted_results[method][driver] = int(drivers_results[idx2, idx1])

    # Calculate average rankings across all methods
    formatted_results['AVG'] = {}
    # Convert to float32 for averaging, then compute mean across columns (axis=1)
    avg_ranking = np.mean(drivers_results.astype(np.float32), axis=1)

    # Store the average ranking for each driver as an integer
    for idx, driver in enumerate(ordered_drivers):
        formatted_results['AVG'][driver] = int(avg_ranking[idx])

    return formatted_results


def rank_test(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    ranking_params: params.RankingParams,
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ]
    | None = None,
    max_lag: int = 0,
) -> dict[
    dts.AccountType,
    dict[dts.DriverClassification, dict[str, dict[dts.DriverName, int]]],
]:
    """
    Rank drivers for testing purposes across multiple accounts and classifications.

    This function executes selected ranking methods for each combination of account
    type and driver classification, applying lag transformations as needed. It returns
    integer rankings organized by account type, driver classification, and method.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Object containing account groups and classified driver groups for analysis.
    ranking_params : params.RankingParams
        Configuration parameters for the ranking methods.
    best_lags : dict[dts.AccountType, dict[dts.DriverClassification,
        dict[dts.DriverName, int]]] | None, optional
        Dictionary mapping account types, driver classifications, and driver names
        to their optimal lag values. If None, all lags default to 0.
    max_lag : int, default=0
        Maximum lag value to apply to the data.

    Returns
    -------
    dict[dts.AccountType, dict[dts.DriverClassification, dict[str, dict[dts.DriverName, int]]]]
        Nested dictionary containing integer rankings organized by:
        - Account type (outer key)
        - Driver classification (second level key)
        - Method name (third level key)
        - Driver name to ranking value (innermost dict)
    """
    # Determine which ranking methods to use based on configuration
    selected_methods = select_methods(methods_params=ranking_params.methods)

    # Ensure at least one method is selected to perform rankings
    assert len(selected_methods) > 0, (
        'Must use at least one Ranking method, check config file.'
    )

    # Initialize best_lags if not provided - set all lags to 0 by default
    if best_lags is None:
        best_lags = {}
        # Iterate through all account types
        for account_type in accounts_drivers_info.accounts.account_map.keys():
            best_lags[account_type] = {}
            # Iterate through all driver classifications (e.g., positive, negative, neutral)
            for (
                driver_classification
            ) in accounts_drivers_info.classified_drivers.classification_groups.keys():
                best_lags[account_type][driver_classification] = {}
                # Get the classification index for accessing the correct driver map
                class_idx = (
                    accounts_drivers_info.classified_drivers.classification_groups[
                        driver_classification
                    ]
                )
                # Set lag to 0 for each driver in this classification
                for driver in accounts_drivers_info.classified_drivers.maps[
                    class_idx
                ].keys():
                    best_lags[account_type][driver_classification][driver] = 0

    # Initialize nested dictionary structure to store integer rankings by
    # account type and driver classification
    method_values: dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[str, dict[dts.DriverName, int]]],
    ] = {}

    # Iterate through each account type in the account driver info
    for account in accounts_drivers_info.accounts.account_map.keys():
        # Initialize dictionary for current account type
        method_values[account] = {}

        # Iterate through each driver classification (e.g., positive, negative, neutral)
        for (
            driver_classification
        ) in accounts_drivers_info.classified_drivers.classification_groups.keys():
            # Create AccountDriverGroup with lag-adjusted account and driver data
            account_driver_group = dts.AccountDriverGroup(
                account=accounts_drivers_info.accounts[account].apply_lag(
                    max_lag=max_lag
                ),
                drivers=accounts_drivers_info.classified_drivers[
                    driver_classification
                ].apply_lags(
                    best_lags=best_lags[account][driver_classification],
                    max_lag=max_lag,
                ),
                np_dtype=accounts_drivers_info.np_dtype,
            )

            # Run all selected ranking methods for the current account-classification combination
            # Returns array of integer rankings with shape (num_drivers, num_methods)
            drivers_results: ArrayI = run_methods(
                info=account_driver_group,
                selected_methods=selected_methods,
                ranking_params=ranking_params,
            )

            # Format the raw ranking results into a structured dictionary
            # organized by method name and driver name with integer rankings
            method_values[account][driver_classification] = _format_results_int(
                drivers_results=drivers_results,
                ordered_drivers=accounts_drivers_info.classified_drivers.get_ordered_drivers(
                    classification=driver_classification
                ),
                selected_methods=selected_methods,
            )

    return method_values


def rank_with_details(
    account_driver_info: dts.AccountDriverGroup,
    ranking_params: params.RankingParams,
) -> tuple[dict[dts.DriverName, int], list[dict[dts.DriverName, float]], list[str]]:
    """
    Ranks drivers and returns individual method rankings along with final averaged rankings.

    Parameters
    ----------
    account_driver_info : dts.AccountDriverGroup
        Object containing the driver and account data needed for analysis.
    ranking_params : params.RankingParams
        Configuration parameters for the ranking methods.
    testing_params : params.TestingParams
        Parameters for testing and debugging purposes.

    Returns
    -------
    tuple[dict[dts.DriverName, int], list[dict[dts.DriverName, float]], list[str]]
        A tuple containing:
        - Final integer rankings for each driver (averaged across methods)
        - List of ranking dictionaries from each individual method
        - List of method names corresponding to the rankings
    """
    # Select which ranking methods to use
    selected_methods = select_methods(methods_params=ranking_params.methods)

    # Get average rankings and individual method rankings
    method_instances: list[AbstractRankingMethod] = []
    for method in selected_methods:
        method_instances.append(
            RANKING_METHOD_MAP[method](account_driver_info, ranking_params[method])
        )

    method_rankings = []
    method_names: list[str] = []

    for method in method_instances:
        method_rankings.append(method.rank())
        method_names.append(method.name())

    # Calculate average rankings
    avg_rankings: dict[dts.DriverName, float] = {}
    for driver in account_driver_info.drivers.map.keys():
        arr_driver_rankings = np.zeros((len(selected_methods)))
        for i, ranking in enumerate(method_rankings):
            arr_driver_rankings[i] = ranking[driver]
        avg_rankings[driver] = float(arr_driver_rankings.mean())

    # Convert to integer ranks
    arr_avg_rankings = np.zeros((account_driver_info.drivers.arr.shape[0],))
    drivers_map_flipped = account_driver_info.drivers.flip_map()
    for k in range(arr_avg_rankings.shape[0]):
        arr_avg_rankings[k] = avg_rankings[drivers_map_flipped[k]]

    final_rankings = rank_array_to_ranking_dict(
        rank_array=order_drivers_no_ties(arr_avg_rankings=arr_avg_rankings),
        ordered_drivers=account_driver_info.drivers.get_ordered_drivers(),
    )

    return final_rankings, method_rankings, method_names


def _format_results_float(
    drivers_results: ArrayF,
    ordered_drivers: Sequence[dts.DriverName],
    selected_methods: Sequence[rdt.DriverRankingEnum],
) -> dict[str, dict[dts.DriverName, float]]:
    """
    Format driver ranking results into a structured dictionary with float values.

    This function organizes the float ranking results from multiple methods
    into a nested dictionary structure for easier access and analysis. It sorts
    results by method values, computes average rankings across methods, and
    generates a final ranking.

    Parameters
    ----------
    drivers_results : ArrayF
        2D array of float rankings where rows correspond to drivers and columns
        correspond to different ranking methods.
    ordered_drivers : Sequence[dts.DriverName]
        Sequence of driver names in the same order as the rows in drivers_results.
    selected_methods : Sequence[rdt.DriverRankingEnum]
        Ranking methods that were used, corresponding to columns in drivers_results.

    Returns
    -------
    dict[str, dict[dts.DriverName, float]]
        Formatted ranking results organized by method name and driver name.
        Each method key maps to a dictionary of driver names to their float values,
        sorted in descending order by value. Includes a 'Final Rank' key containing
        the consolidated integer rankings based on averaged method ranks.
    """
    # Get the string names for each ranking method from the method map
    method_names: list[str] = [
        RANKING_METHOD_MAP[method].name() for method in selected_methods
    ]

    # Initialize dictionary to store rankings for each method and driver
    formatted_results: dict[str, dict[dts.DriverName, float]] = {}

    # Iterate through each ranking method
    for i, method in enumerate(method_names):
        formatted_results[method] = {}

        # Sort drivers by their method values in descending order (highest values first)
        # np.argsort with negative values reverses the sort order
        sorted_indicies = np.argsort(-drivers_results[:, i])

        # Store each driver's value for this method, sorted by descending value
        for idx in sorted_indicies:
            formatted_results[method][ordered_drivers[idx]] = float(
                drivers_results[idx, i]
            )

    # Convert continuous method values to integer ranks for each method
    # Shape: (num_drivers, num_methods)
    ranks_arr: ArrayI = np.zeros(drivers_results.shape, dtype=np.int32)
    for i in range(drivers_results.shape[1]):
        # Convert each method's continuous values to ranks (allows ties)
        ranks_arr[:, i] = order_drivers_allow_ties(
            values=drivers_results[:, i],
        )

    # Calculate average rank across all methods for each driver
    # Convert to float32 for averaging, then compute mean across columns (axis=1)
    avg_ranking = np.mean(ranks_arr.astype(np.float32), axis=1)

    # Convert average ranks to final integer rankings (no ties allowed)
    rankings_final = rank_array_to_ranking_dict(
        rank_array=order_drivers_no_ties(arr_avg_rankings=avg_ranking),
        ordered_drivers=ordered_drivers,
    )

    # Add final rankings to results, sorted by rank (best to worst)
    formatted_results['Final Rank'] = {}
    for key in sorted(rankings_final.keys(), key=lambda x: rankings_final[x]):
        formatted_results['Final Rank'][key] = float(rankings_final[key])

    return formatted_results


def _run_methods_debug(
    info: dts.AccountDriverGroup,
    selected_methods: Sequence[rdt.DriverRankingEnum],
    ranking_params: params.RankingParams,
) -> ArrayF:
    """
    Execute ranking methods in debug mode with timing and error tracking.

    This function runs each selected ranking method sequentially, collecting
    float-valued rankings for each driver. It includes performance timing
    for each method and enhanced error reporting to identify which method
    fails if an exception occurs.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        Object containing the driver and account data needed for analysis.
    selected_methods : Sequence[rdt.DriverRankingEnum]
        Sequence of ranking method enums to execute.
    ranking_params : params.RankingParams
        Configuration parameters for the ranking methods.

    Returns
    -------
    ArrayF
        2D array of float values where rows correspond to drivers and columns
        correspond to different ranking methods. Shape is (num_drivers, num_methods).

    Raises
    ------
    Exception
        Re-raises any exception that occurs during method execution, with an
        added note identifying which method caused the error.

    Notes
    -----
    This function logs timing information for each method execution using
    the module logger. It is intended for debugging and performance analysis.
    """
    # Initialize list to store ranking method instances
    method_instances: list[AbstractRankingMethod] = []

    # Create instances of each selected ranking method
    for method in selected_methods:
        method_instances.append(
            RANKING_METHOD_MAP[method](info, ranking_params[method])
        )

    # Array to store rankings from each method
    method_values: ArrayF = np.zeros(
        (info.drivers.arr.shape[0], len(selected_methods)),
        dtype=info.np_dtype,
    )

    # Start timing for performance measurement
    last_time = time.time()

    # Execute each ranking method and collect results
    for idx, method in enumerate(method_instances):
        try:
            method_values[:, idx] = method.apply()
        except Exception as exc:
            # Identify which method caused the error
            exc.add_note(f'Method that caused the problem - {method.name()}')
            raise exc

        logger.timing(
            f'Time of method {method.name()} - {time.time() - last_time:.4f} seconds.'
        )
        last_time = time.time()

    return method_values


def rank_debug(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    ranking_params: params.RankingParams,
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ]
    | None = None,
    max_lag: int = 0,
) -> dict[
    dts.AccountType,
    dict[dts.DriverClassification, dict[str, dict[dts.DriverName, float]]],
]:
    """
    Rank drivers in debug mode with detailed timing and float-valued results.

    This function executes selected ranking methods for each combination of account
    type and driver classification in debug mode, applying lag transformations as needed.
    It returns float-valued rankings organized by account type, driver classification,
    and method, along with timing information for performance analysis.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Object containing account groups and classified driver groups for analysis.
    ranking_params : params.RankingParams
        Configuration parameters for the ranking methods.
    best_lags : dict[dts.AccountType, dict[dts.DriverClassification,
        dict[dts.DriverName, int]]] | None, optional
        Dictionary mapping account types, driver classifications, and driver names
        to their optimal lag values. If None, all lags default to 0.
    max_lag : int, default=0
        Maximum lag value to apply to the data.

    Returns
    -------
    dict[dts.AccountType, dict[dts.DriverClassification, dict[str, dict[dts.DriverName, float]]]]
        Nested dictionary containing float-valued rankings organized by:
        - Account type (outer key)
        - Driver classification (second level key)
        - Method name (third level key), including 'Final Rank'
        - Driver name to ranking value (innermost dict)

    Notes
    -----
    This function uses `_run_methods_debug` which logs timing information for each
    method execution and provides enhanced error reporting. The results include both
    individual method rankings and a consolidated 'Final Rank' based on averaged ranks.
    """
    # Determine which ranking methods to use based on configuration
    selected_methods = select_methods(methods_params=ranking_params.methods)
    # Ensure at least one method is selected to perform rankings
    assert len(selected_methods) > 0, (
        'Must use at least one Ranking method, check config file.'
    )

    if best_lags is None:
        best_lags = {}
        for account_type in accounts_drivers_info.accounts.account_map.keys():
            best_lags[account_type] = {}
            for (
                driver_classification
            ) in accounts_drivers_info.classified_drivers.classification_groups.keys():
                best_lags[account_type][driver_classification] = {}
                class_idx = (
                    accounts_drivers_info.classified_drivers.classification_groups[
                        driver_classification
                    ]
                )
                for driver in accounts_drivers_info.classified_drivers.maps[
                    class_idx
                ].keys():
                    best_lags[account_type][driver_classification][driver] = 0

    # Initialize nested dictionary structure to store rankings by
    # account type and driver classification
    method_values: dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[str, dict[dts.DriverName, float]]],
    ] = {}

    # Iterate through each account type in the account driver info
    for account in accounts_drivers_info.accounts.account_map.keys():
        # Initialize dictionary for current account type
        method_values[account] = {}
        # Iterate through each driver classification (e.g., positive, negative, neutral)
        for (
            driver_classification
        ) in accounts_drivers_info.classified_drivers.classification_groups.keys():
            # Run all selected ranking methods for the
            # current account-classification combination
            info = dts.AccountDriverGroup(
                account=accounts_drivers_info.accounts[account].apply_lag(
                    max_lag=max_lag
                ),
                drivers=accounts_drivers_info.classified_drivers[
                    driver_classification
                ].apply_lags(
                    best_lags=best_lags[account][driver_classification],
                    max_lag=max_lag,
                ),
                np_dtype=accounts_drivers_info.np_dtype,
            )
            drivers_results_arr: ArrayF = _run_methods_debug(
                info=info,
                selected_methods=selected_methods,
                ranking_params=ranking_params,
            )
            # Format and store the results
            method_values[account][driver_classification] = _format_results_float(
                drivers_results=drivers_results_arr,
                ordered_drivers=info.drivers.get_ordered_drivers(),
                selected_methods=selected_methods,
            )
    return method_values

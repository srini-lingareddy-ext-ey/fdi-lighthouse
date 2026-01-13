import time
from typing import Optional, Sequence

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params as params
from lh_v2.shared import ArrayF, ArrayI
from lh_v2.util import CustomLogger, get_logger

from .driver_ranking_methods import (
    AbstractRankingMethod,
)
from .driver_ranking_shared import RANKING_METHOD_MAP, select_methods
from .driver_ranking_util import (
    order_drivers_no_ties,
    rank_array_to_ranking_dict,
    set_lags_to_zero,
)

logger = get_logger(__name__)


def run_methods(
    info: dts.AccountDriverGroup,
    selected_methods: Sequence[rdt.DriverRankingEnum],
    ranking_params: params.RankingParams,
    logger: Optional[CustomLogger] = logger,
) -> ArrayI:
    """
    Execute all selected driver ranking methods and collect their results.

    Instantiates and runs each ranking method specified in `selected_methods`,
    collecting the ranking scores for each driver from every method. Returns
    a dictionary mapping each driver name to an array of ranking scores
    (one score per method).

    Timing information for each method is logged if enabled.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        Object containing the driver and account data needed for analysis.
    selected_methods : Sequence[rdt.DriverRankingEnum]
        Ranking methods to be used.
    ranking_params : params.RankingParams
        Configuration parameters for the ranking methods.

    Returns
    -------
    dict[dts.DriverName, ArrayF]
        Dictionary mapping driver names to arrays of ranking scores,
        where each array contains the scores from all selected methods.

    Raises
    ------
    Exception
        Propagates any exception raised by individual ranking methods,
        with an additional note indicating which method caused the error.
    """

    if logger is None:
        logger = get_logger(__name__)

    # Initialize list to store ranking method instances
    method_instances: list[AbstractRankingMethod] = []

    # Create instances of each selected ranking method
    for method in selected_methods:
        method_instances.append(
            RANKING_METHOD_MAP[method](info, ranking_params[method])
        )

    # List to store rankings from each method
    method_rankings: list[dict[dts.DriverName, int]] = []

    # Start timing for performance measurement
    last_time = time.time()

    # Execute each ranking method and collect results
    for method in method_instances:
        try:
            method_rankings.append(method.rank())
        except Exception as exc:
            # Identify which method caused the error
            exc.add_note(f'Method that caused the problem - {method.name()}')
            raise exc

        logger.timing(
            f'Time of method `{method.name()}` - {time.time() - last_time:.4f} seconds.'
        )
        last_time = time.time()

    method_rankings_arr: ArrayI = np.zeros(
        (info.drivers.arr.shape[0], len(selected_methods)),
        dtype=np.int32,
    )

    # Collect ranking scores for each driver across all methods
    for idx, driver in enumerate(info.drivers.get_ordered_drivers()):
        for i, ranking in enumerate(method_rankings):
            method_rankings_arr[idx, i] = ranking[driver]

    return method_rankings_arr


def rank(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    ranking_params: params.RankingParams,
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ]
    | None = None,
    max_lag: int = 0,
    logger: Optional[CustomLogger] = None,
) -> dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]]:
    """
    Compute driver rankings across multiple statistical methods.

    This function ranks drivers for each account type and driver classification
    using a configurable set of statistical and machine learning methods.
    It aggregates the results by averaging the scores from all selected methods
    and converts these averages into integer ranks.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Object containing account and classified driver data for analysis.
    ranking_params : params.RankingParams
        Configuration parameters specifying which ranking methods to use and their settings.
    best_lags : dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]], optional
        Dictionary specifying the best lag for each driver, grouped by account type and classification.
        If None, all lags are set to zero.
    max_lag : int, default=0
        Maximum lag to apply to account and driver data.

    Returns
    -------
    dict
        Nested dictionary mapping account type and driver classification to driver ranks:
        {account_type: {driver_classification: {driver_name: rank}}}
    """
    if logger is None:
        logger = get_logger(__name__)
    logger.info('Ranking drivers.')

    # Select ranking methods based on configuration
    selected_methods = select_methods(methods_params=ranking_params.methods)
    # Ensure at least one ranking method is selected
    assert len(selected_methods) > 0, (
        'Must use at least one Ranking method, check config file.'
    )

    # If best_lags is not provided, initialize all lags to zero for each driver
    if best_lags is None:
        best_lags = set_lags_to_zero(accounts_drivers_info=accounts_drivers_info)

    # Initialize output dictionary for rankings
    rankings: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ] = {}

    # Loop over each account type
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        rankings[account] = {}
        # Loop over each driver classification (e.g., positive, negative, neutral)
        for (
            class_
        ) in accounts_drivers_info.classified_drivers.get_ordered_classifications():
            # Prepare account and driver data with appropriate lags
            info = dts.AccountDriverGroup(
                account=accounts_drivers_info.accounts[account].apply_lag(
                    max_lag=max_lag,
                ),
                drivers=accounts_drivers_info.classified_drivers[class_].apply_lags(
                    best_lags=best_lags[account][class_],
                    max_lag=max_lag,
                ),
                np_dtype=accounts_drivers_info.np_dtype,
            )
            drivers_results: ArrayI = run_methods(
                info=info,
                selected_methods=selected_methods,
                ranking_params=ranking_params,
                logger=logger,
            )

            avg_rankings_arr: ArrayF = np.mean(
                drivers_results.astype(np.float32), axis=1
            )

            # Convert average scores to integer ranks (lowest score = rank 1)
            rankings[account][class_] = rank_array_to_ranking_dict(
                rank_array=order_drivers_no_ties(
                    arr_avg_rankings=avg_rankings_arr,
                ),
                ordered_drivers=info.drivers.get_ordered_drivers(),
            )

    # Return nested dictionary of rankings
    return rankings

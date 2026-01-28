from typing import Optional

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params as params
from lh_v2.shared import ArrayF, ArrayI
from lh_v2.util import CustomLogger, get_logger

from .driver_ranking_shared import RANKING_METHOD_MAP, select_methods
from .driver_ranking_util import (
    order_drivers_no_ties,
    set_lags_to_zero,
)
from .rank_drivers import run_methods


def rank_llm(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    ranking_params: params.RankingParams,
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ]
    | None = None,
    max_lag: int = 0,
    logger: Optional[CustomLogger] = None,
) -> dict[
    dts.AccountType,
    dict[
        dts.DriverClassification,
        dict[dts.DriverName, dict[rdt.DriverRankingLLMMetric, float]],
    ],
]:
    """
    Compute comprehensive driver rankings with multiple metrics for LLM-based selection.

    This function ranks drivers for each account type and driver classification
    using a configurable set of statistical and machine learning methods. Unlike
    the standard ranking function, this returns detailed metrics for each driver
    to support downstream LLM-based driver selection.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Object containing account and classified driver data for analysis.
    ranking_params : params.RankingParams
        Configuration parameters specifying which ranking methods to use and their
        settings, including which LLM metrics to compute.
    best_lags : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]], optional
        Dictionary specifying the optimal lag for each driver, organized by account
        type and classification. If None, all lags are set to zero.
    max_lag : int, default=0
        Maximum lag value to apply to account and driver data before analysis.
    logger : Optional[CustomLogger], default=None
        Logger instance for tracking progress. If None, a default logger is created.

    Returns
    -------
    dict[AccountType, dict[DriverClassification, dict[DriverName, dict[DriverRankingLLMMetric, float]]]]
        Nested dictionary containing comprehensive metrics for each driver, organized by:
        - Account type (outer key)
        - Driver classification (second level key)
        - Driver name (third level key)
        - Metric type to metric value (innermost dict)

        Metrics can include:
        - FINAL_RANK: The final consolidated ranking position
        - AVG_RANK: The average rank across all methods
        - Individual method scores (e.g., PEARSON_CORRELATION, LASSO, etc.)

    Raises
    ------
    AssertionError
        If no ranking methods are selected in the configuration.

    Notes
    -----
    The function applies lags to both account and driver data before ranking,
    ensuring temporal alignment. The specific metrics returned depend on the
    llm_metrics configuration in ranking_params.

    See Also
    --------
    rank : Standard ranking function that returns only final ranks.
    rank_debug : Debug ranking function with full method details.

    Examples
    --------
    >>> metrics = rank_llm(
    ...     accounts_drivers_info=data,
    ...     ranking_params=params,
    ...     best_lags=lags,
    ...     max_lag=12
    ... )
    >>> # Access specific metrics
    >>> revenue_metrics = metrics[AccountType('REVENUE')]
    >>> primary_drivers = revenue_metrics[DriverClassification.PRIMARY]
    >>> driver1_rank = primary_drivers['driver1'][DriverRankingLLMMetric.FINAL_RANK]
    """
    # Create logger if not provided
    if logger is None:
        logger = get_logger(__name__)
    logger.info('Ranking drivers.')

    # Select ranking methods based on configuration parameters
    selected_methods = select_methods(methods_params=ranking_params.methods)
    # Validate that at least one method is selected to perform ranking
    assert len(selected_methods) > 0, (
        'Must use at least one Ranking method, check config file.'
    )

    # Initialize all driver lags to zero if not provided
    # This creates a baseline where no temporal offset is applied
    if best_lags is None:
        best_lags = set_lags_to_zero(accounts_drivers_info=accounts_drivers_info)

    # Initialize nested dictionary to store comprehensive metrics for all drivers
    ranking_metrics: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingLLMMetric, float]],
        ],
    ] = {}

    # Iterate through each account type to perform independent ranking
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        ranking_metrics[account] = {}

        # Iterate through each driver classification within this account
        # (e.g., PRIMARY, SECONDARY, etc.)
        for (
            class_
        ) in accounts_drivers_info.classified_drivers.get_ordered_classifications():
            # Create AccountDriverGroup with lagged data for temporal alignment
            # Account data is lagged by max_lag, drivers by their individual optimal lags
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

            # Execute all selected ranking methods and collect results
            # Returns a 2D array: (n_drivers, n_methods) with ranks from each method
            drivers_results: ArrayI = run_methods(
                info=info,
                selected_methods=selected_methods,
                ranking_params=ranking_params,
                logger=logger,
            )

            # Calculate average rank across all methods for each driver
            # This provides a consolidated performance metric
            avg_rankings_arr: ArrayF = np.mean(
                drivers_results.astype(np.float32), axis=1
            )

            # Initialize dictionary for this account-classification combination
            ranking_metrics[account][class_] = {}

            # Convert average ranks to final integer rankings without ties
            # Lower average rank leads to better final ranking (rank 1 is best)
            final_rankings_arr = order_drivers_no_ties(
                arr_avg_rankings=avg_rankings_arr
            )

            # Initialize metric dictionaries for each driver
            for driver in info.drivers.get_ordered_drivers():
                ranking_metrics[account][class_][driver] = {}

            # Populate requested metrics for each driver based on configuration
            for metric in ranking_params.llm_ranking_params.llm_metrics:
                match metric:
                    case rdt.DriverRankingLLMMetric.FINAL_RANK:
                        # Store the final consolidated rank for each driver
                        for idx, driver in enumerate(
                            info.drivers.get_ordered_drivers()
                        ):
                            ranking_metrics[account][class_][driver][
                                rdt.DriverRankingLLMMetric.FINAL_RANK
                            ] = float(final_rankings_arr[idx])

                    case rdt.DriverRankingLLMMetric.AVG_RANK:
                        # Store the average rank across all methods for each driver
                        for idx, driver in enumerate(
                            info.drivers.get_ordered_drivers()
                        ):
                            ranking_metrics[account][class_][driver][
                                rdt.DriverRankingLLMMetric.AVG_RANK
                            ] = float(avg_rankings_arr[idx])

                    case _:
                        # Handle individual method metrics (e.g., PEARSON_CORRELATION, LASSO)
                        # Instantiate the specific ranking method and compute its scores
                        method_instance = RANKING_METHOD_MAP[
                            rdt.DriverRankingEnum(metric.value)
                        ](info, ranking_params[rdt.DriverRankingEnum(metric.value)])
                        method_val_arr = method_instance.apply()

                        # Store the method-specific score for each driver
                        for idx, driver in enumerate(
                            info.drivers.get_ordered_drivers()
                        ):
                            ranking_metrics[account][class_][driver][metric] = float(
                                method_val_arr[idx]
                            )

    # Return comprehensive metrics for LLM-based driver selection
    return ranking_metrics

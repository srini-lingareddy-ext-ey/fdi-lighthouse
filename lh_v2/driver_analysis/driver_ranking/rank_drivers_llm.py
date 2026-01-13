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

    ranking_metrics: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingLLMMetric, float]],
        ],
    ] = {}

    # Loop over each account type
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        ranking_metrics[account] = {}
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

            ranking_metrics[account][class_] = {}

            # Convert average scores to integer ranks (lowest score = rank 1)
            final_rankings_arr = order_drivers_no_ties(
                arr_avg_rankings=avg_rankings_arr
            )

            for driver in info.drivers.get_ordered_drivers():
                ranking_metrics[account][class_][driver] = {}

            for metric in ranking_params.llm_ranking_params.llm_metrics:
                match metric:
                    case rdt.DriverRankingLLMMetric.FINAL_RANK:
                        for idx, driver in enumerate(
                            info.drivers.get_ordered_drivers()
                        ):
                            ranking_metrics[account][class_][driver][
                                rdt.DriverRankingLLMMetric.FINAL_RANK
                            ] = float(final_rankings_arr[idx])
                    case rdt.DriverRankingLLMMetric.AVG_RANK:
                        for idx, driver in enumerate(
                            info.drivers.get_ordered_drivers()
                        ):
                            ranking_metrics[account][class_][driver][
                                rdt.DriverRankingLLMMetric.AVG_RANK
                            ] = float(avg_rankings_arr[idx])
                    case _:
                        method_instance = RANKING_METHOD_MAP[
                            rdt.DriverRankingEnum(metric.value)
                        ](info, ranking_params[rdt.DriverRankingEnum(metric.value)])
                        method_val_arr = method_instance.apply()
                        for idx, driver in enumerate(
                            info.drivers.get_ordered_drivers()
                        ):
                            ranking_metrics[account][class_][driver][metric] = float(
                                method_val_arr[idx]
                            )

    # Return nested dictionary of rankings
    return ranking_metrics

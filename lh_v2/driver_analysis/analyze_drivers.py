import time

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params as params
from lh_v2.util import get_logger

from .collinearity import prune_collinearity, remove_collinearity
from .driver_analysis_types import (
    DriverAnalysisInput,
    DriverAnalysisOutput,
    DriverAnalysisOutputLLM,
)
from .driver_ranking import rank, rank_debug, rank_llm, rank_test
from .lagging import select_best_lags

logger = get_logger(__name__)


def analyze_drivers_full(
    accounts_drivers_info: DriverAnalysisInput,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
) -> DriverAnalysisOutput:
    """
    Analyze drivers across all accounts by optimizing lags, ranking, and removing collinearity.

    This function performs a complete driver analysis pipeline:
    1. Optimizes lag values for each driver
    2. Ranks drivers based on their performance
    3. Removes collinear drivers to ensure independence

    Parameters
    ----------
    accounts_drivers_info : DriverAnalysisInput
        Input data containing account and driver information.
    lh_params : params.LighthouseParams
        Lighthouse parameters including lag optimization, ranking, collinearity,
        and testing settings.

    Returns
    -------
    DriverAnalysisOutput
        Analysis output containing the original account/driver information,
        selected non-collinear drivers, and optimal lag values.
    """
    logger.info('Starting full driver analysis.')

    # Filter input data to training date range for analysis
    info = accounts_drivers_info.apply_daterange(
        start_date=general_params.training_start_date,
        end_date=general_params.training_end_date,
    )

    start_time = time.time()
    last_time = time.time()

    # Step 1: Find optimal lag for each driver to maximize predictive power
    best_lags = select_best_lags(info=accounts_drivers_info, da_params=da_params)
    logger.timing(
        f'Optimal lag selection completed in {time.time() - last_time:.4f} seconds.'
    )

    last_time = time.time()

    # Step 2: Rank drivers by their contribution to model performance using optimal lags
    driver_rankings = rank(
        accounts_drivers_info=info,
        ranking_params=da_params.ranking_params,
        best_lags=best_lags,
        max_lag=da_params.lag_params.n_max_lag,
    )
    logger.timing(f'Driver ranking completed in {time.time() - last_time:.4f} seconds.')

    last_time = time.time()

    # Step 3: Remove highly correlated drivers, keeping highest-ranked ones to avoid multicollinearity
    selected_drivers = remove_collinearity(
        accounts_drivers_info=info,
        classified_driver_rankings=driver_rankings,
        da_params=da_params,
        best_lags=best_lags,
    )
    logger.timing(
        f'Collinearity removal completed in {time.time() - last_time:.4f} seconds.'
    )
    logger.timing(
        f'Total Driver Analysis time: {time.time() - start_time:.4f} seconds.'
    )

    # Return final set of selected drivers with their optimal lags
    return DriverAnalysisOutput(
        selected_drivers=selected_drivers,
        lags=best_lags,
    )


def analyze_drivers_llm(
    accounts_drivers_info: DriverAnalysisInput,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
):
    logger.info('Starting LLM-based driver analysis.')

    # Filter input data to training date range for analysis
    info = accounts_drivers_info.apply_daterange(
        start_date=general_params.training_start_date,
        end_date=general_params.training_end_date,
    )

    start_time = time.time()
    last_time = time.time()

    # Step 1: Find optimal lag for each driver to maximize predictive power
    best_lags = select_best_lags(info=accounts_drivers_info, da_params=da_params)
    logger.timing(
        f'Optimal lag selection completed in {time.time() - last_time:.4f} seconds.'
    )

    last_time = time.time()

    # Step 2: Rank drivers by their contribution to model performance using optimal lags
    driver_rankings = rank_llm(
        accounts_drivers_info=info,
        ranking_params=da_params.ranking_params,
        best_lags=best_lags,
        max_lag=da_params.lag_params.n_max_lag,
    )
    logger.timing(f'Driver ranking completed in {time.time() - last_time:.4f} seconds.')

    last_time = time.time()

    # Extract total driver ranking from driver_rankings from rank_llm
    total_driver_rankings: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ] = {}
    for account in driver_rankings.keys():
        total_driver_rankings[account] = {}
        for class_ in driver_rankings[account].keys():
            total_driver_rankings[account][class_] = {}
            for driver in driver_rankings[account][class_].keys():
                total_driver_rankings[account][class_][driver] = round(
                    driver_rankings[account][class_][driver][
                        rdt.DriverRankingLLMMetric.FINAL_RANK
                    ]
                )

    # Step 3: Remove highly correlated drivers, keeping highest-ranked ones to avoid multicollinearity
    selectable_drivers = prune_collinearity(
        accounts_drivers_info=info,
        classified_driver_rankings=total_driver_rankings,
        da_params=da_params,
        best_lags=best_lags,
    )
    logger.timing(
        f'Collinearity removal completed in {time.time() - last_time:.4f} seconds.'
    )
    logger.timing(
        f'Total Driver Analysis time: {time.time() - start_time:.4f} seconds.'
    )

    # Create new driver_rankings dict with only selectable drivers
    driver_rankings_filtered: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingLLMMetric, float]],
        ],
    ] = {}
    for account in selectable_drivers.keys():
        driver_rankings_filtered[account] = {}
        for class_ in selectable_drivers[account].keys():
            driver_rankings_filtered[account][class_] = {}

            for driver in selectable_drivers[account][class_]:
                driver_rankings_filtered[account][class_][driver] = driver_rankings[
                    account
                ][class_][driver]

    # Return final set of selectable drivers with their optimal lags and
    # ranking metrics for LLM
    return DriverAnalysisOutputLLM(
        selectable_drivers=selectable_drivers,
        metrics=driver_rankings_filtered,
        lags=best_lags,
    )


def driver_ranking_test(
    accounts_drivers_info: DriverAnalysisInput,
    da_params: params.DriverAnalysisParams,
) -> dict[
    dts.AccountType,
    dict[dts.DriverClassification, dict[str, dict[dts.DriverName, int]]],
]:
    driver_rankings_test = rank_test(
        accounts_drivers_info=accounts_drivers_info,
        ranking_params=da_params.ranking_params,
        max_lag=da_params.lag_params.n_max_lag,
    )

    return driver_rankings_test


def driver_ranking_debug(
    accounts_drivers_info: DriverAnalysisInput,
    da_params: params.DriverAnalysisParams,
) -> tuple[
    dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[str, dict[dts.DriverName, float]]],
    ],
    dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]],
]:
    """
    Debug driver ranking by computing full rankings and optimal lags.

    This function performs lag optimization and driver ranking, returning
    detailed ranking information for debugging purposes.

    Parameters
    ----------
    accounts_drivers_info : DriverAnalysisInput
        Input data containing account and driver information.
    lh_params : params.LighthouseParams
        Lighthouse parameters including lag optimization and ranking settings.

    Returns
    -------
    driver_rankings_full : dict
        Nested dictionary containing full driver ranking scores by account type,
        classification, metric, and driver name.
    best_lags : dict
        Nested dictionary containing optimal lag values by account type,
        classification, and driver name.
    """
    # Find optimal lag for each driver to maximize predictive power
    best_lags = select_best_lags(info=accounts_drivers_info, da_params=da_params)

    # Generate full ranking scores for all drivers using optimal lags
    driver_rankings_full = rank_debug(
        accounts_drivers_info=accounts_drivers_info,
        ranking_params=da_params.ranking_params,
        best_lags=best_lags,
        max_lag=da_params.lag_params.n_max_lag,
    )

    # Return detailed rankings and lags for debugging analysis
    return driver_rankings_full, best_lags

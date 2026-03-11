import time

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params as params
from lh_v2.io.output import (
    save_collinearity_results,
    save_driver_ranking_results,
    save_lag_ranking_results,
    save_selected_drivers_results,
)
from lh_v2.util import get_logger

from .collinearity import (
    get_collinearity_arr,
    prune_final_drivers,
    select_final_drivers,
)
from .driver_analysis_types import (
    DriverAnalysisInput,
    DriverAnalysisOutput,
    DriverAnalysisOutputLLM,
)
from .driver_ranking import (
    handle_ranking_metrics,
    rank,
)
from .lagging import select_best_lags

logger = get_logger(__name__)


def analyze_drivers_full(
    accounts_drivers_info: DriverAnalysisInput,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
    output_params: params.OutputParams,
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
    general_params : params.GeneralParams
        General parameters including training date range settings.
    da_params : params.DriverAnalysisParams
        Driver analysis parameters including lag optimization, ranking,
        and collinearity settings.

    Returns
    -------
    DriverAnalysisOutput
        Analysis output containing the selected non-collinear drivers and
        optimal lag values for each driver.

    Notes
    -----
    The function filters data to the training date range before analysis.
    Timing information for each step is logged for performance monitoring.

    See Also
    --------
    analyze_drivers_llm : LLM-based variant of driver analysis.
    select_best_lags : Function for lag optimization.
    rank : Function for driver ranking.
    remove_collinearity : Function for collinearity removal.
    """
    logger.info('Starting full driver analysis.')

    # Filter input data to training date range for analysis
    info = accounts_drivers_info.apply_daterange(
        start_date=general_params.training_start_date,
        end_date=general_params.training_end_date,
    )

    classification_map = info.classified_drivers.get_driver_classification_map()

    start_time = time.time()
    last_time = time.time()

    # Step 1: Find optimal lag for each driver to maximize predictive power
    best_lags, lagging_ranking_results = select_best_lags(
        info=info, da_params=da_params
    )
    save_lag_ranking_results(
        lag_ranking_results=lagging_ranking_results,
        classification_map=classification_map,
        best_lags=best_lags,
        output_params=output_params,
        lag_params=da_params.lag_params,
    )

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
    driver_ranking_metrics = handle_ranking_metrics(
        accounts_drivers_info=info,
        ranking_params=da_params.ranking_params,
        dict_method_values=driver_rankings,
        best_lags=best_lags,
    )
    save_driver_ranking_results(
        driver_ranking_results=driver_rankings,
        best_lags=best_lags,
        output_params=output_params,
    )
    logger.timing(f'Driver ranking completed in {time.time() - last_time:.4f} seconds.')

    last_time = time.time()

    final_driver_rankings: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ] = {}
    for account in driver_rankings.keys():
        final_driver_rankings[account] = {}
        for class_ in driver_rankings[account].keys():
            final_driver_rankings[account][class_] = {}
            for driver in driver_rankings[account][class_].keys():
                # Round the final rank to an integer for discrete ranking
                final_driver_rankings[account][class_][driver] = round(
                    driver_rankings[account][class_][driver][
                        rdt.DriverRankingMetric.FINAL_RANK
                    ]
                )

    # Step 3: Remove highly correlated drivers, keeping highest-ranked ones to avoid multicollinearity
    combined_drivers = info.classified_drivers.combine()
    flattened_lags = _flatten_drivers_lags(lags=best_lags)
    arr_collinearity_dict = get_collinearity_arr(
        acc_lst=info.accounts.get_ordered_accounts(),
        drivers_combined=combined_drivers,
        da_params=da_params,
        flattened_lags=flattened_lags,
    )

    preselected_drivers = {}
    for account in info.accounts.get_ordered_accounts():
        preselected_drivers[account] = {}
        for class_ in info.classified_drivers.get_ordered_classifications():
            preselected_drivers[account][class_] = []

    selected_drivers, skipped_drivers = select_final_drivers(
        drivers_map=combined_drivers.map,
        classified_driver_rankings=final_driver_rankings,
        da_params=da_params,
        preselected_drivers=preselected_drivers,
        arr_collinearity_dict=arr_collinearity_dict,
    )

    save_collinearity_results(
        dict_collinearity_arr=arr_collinearity_dict,
        ordered_drivers=combined_drivers.get_ordered_drivers(),
        output_params=output_params,
    )

    logger.timing(
        f'Collinearity removal completed in {time.time() - last_time:.4f} seconds.'
    )

    save_selected_drivers_results(
        driver_rankings=final_driver_rankings,
        selected_drivers=selected_drivers,
        skipped_drivers=skipped_drivers,
        output_params=output_params,
    )

    logger.timing(
        f'Total Driver Analysis time: {time.time() - start_time:.4f} seconds.'
    )

    # Return final set of selected drivers with their optimal lags
    return DriverAnalysisOutput(
        selected_drivers=selected_drivers,
        metrics=driver_ranking_metrics,
        lags=best_lags,
    )


def analyze_drivers_llm(
    accounts_drivers_info: DriverAnalysisInput,
    general_params: params.GeneralParams,
    da_params: params.DriverAnalysisParams,
) -> DriverAnalysisOutputLLM:
    """
    Analyze drivers using LLM-based ranking with lag optimization and collinearity pruning.

    This function performs driver analysis with LLM-enhanced ranking:
    1. Optimizes lag values for each driver
    2. Ranks drivers using LLM-based metrics
    3. Prunes collinear drivers while preserving ranking information

    Parameters
    ----------
    accounts_drivers_info : DriverAnalysisInput
        Input data containing account and driver information.
    general_params : params.GeneralParams
        General parameters including training date range settings.
    da_params : params.DriverAnalysisParams
        Driver analysis parameters including lag optimization, ranking,
        and collinearity settings.

    Returns
    -------
    DriverAnalysisOutputLLM
        Analysis output containing:
        - selectable_drivers: Non-collinear drivers available for selection
        - metrics: LLM ranking metrics for each selectable driver
        - lags: Optimal lag values for each driver

    Notes
    -----
    Unlike analyze_drivers_full, this function returns ranking metrics alongside
    the selectable drivers, allowing for LLM-based driver selection downstream.
    The function uses prune_collinearity instead of remove_collinearity to
    retain ranking information.

    See Also
    --------
    analyze_drivers_full : Standard driver analysis without LLM ranking.
    rank_llm : LLM-based driver ranking function.
    prune_collinearity : Collinearity pruning that preserves rankings.
    """
    logger.info('Starting LLM-based driver analysis.')

    # Filter input data to training date range for analysis
    info = accounts_drivers_info.apply_daterange(
        start_date=general_params.training_start_date,
        end_date=general_params.training_end_date,
    )

    # Initialize timing variables to track performance of each analysis step
    start_time = time.time()
    last_time = time.time()

    # Step 1: Find optimal lag for each driver to maximize predictive power
    best_lags, _ = select_best_lags(info=accounts_drivers_info, da_params=da_params)
    logger.timing(
        f'Optimal lag selection completed in {time.time() - last_time:.4f} seconds.'
    )

    last_time = time.time()

    # Step 2: Rank drivers using LLM-based metrics that evaluate multiple dimensions
    # Returns detailed metrics for each driver to support downstream LLM selection
    driver_rankings = rank(
        accounts_drivers_info=info,
        ranking_params=da_params.ranking_params,
        best_lags=best_lags,
        max_lag=da_params.lag_params.n_max_lag,
    )
    driver_ranking_metrics = handle_ranking_metrics(
        accounts_drivers_info=info,
        ranking_params=da_params.ranking_params,
        dict_method_values=driver_rankings,
        best_lags=best_lags,
    )
    logger.timing(f'Driver ranking completed in {time.time() - last_time:.4f} seconds.')

    last_time = time.time()

    # Extract the final rank metric from the comprehensive LLM ranking results
    # This creates a simplified ranking structure for collinearity analysis
    total_driver_rankings: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ] = {}
    for account in driver_rankings.keys():
        total_driver_rankings[account] = {}
        for class_ in driver_rankings[account].keys():
            total_driver_rankings[account][class_] = {}
            for driver in driver_rankings[account][class_].keys():
                # Round the final rank to an integer for discrete ranking
                total_driver_rankings[account][class_][driver] = round(
                    driver_rankings[account][class_][driver][
                        rdt.DriverRankingMetric.FINAL_RANK
                    ]
                )

    # Step 3: Prune highly correlated drivers while keeping highest-ranked ones
    # Uses prune_collinearity (not remove_collinearity) to preserve all ranking info
    combined_drivers = info.classified_drivers.combine()
    flattened_lags = _flatten_drivers_lags(lags=best_lags)
    arr_collinearity_dict = get_collinearity_arr(
        acc_lst=info.accounts.get_ordered_accounts(),
        drivers_combined=combined_drivers,
        da_params=da_params,
        flattened_lags=flattened_lags,
    )

    preselected_drivers = {}
    for account in info.accounts.get_ordered_accounts():
        preselected_drivers[account] = {}
        for class_ in info.classified_drivers.get_ordered_classifications():
            preselected_drivers[account][class_] = []

    selectable_drivers = prune_final_drivers(
        drivers_map=combined_drivers.map,
        classified_driver_rankings=total_driver_rankings,
        da_params=da_params,
        preselected_drivers=preselected_drivers,
        arr_collinearity_dict=arr_collinearity_dict,
    )
    logger.timing(
        f'Collinearity removal completed in {time.time() - last_time:.4f} seconds.'
    )
    logger.timing(
        f'Total Driver Analysis time: {time.time() - start_time:.4f} seconds.'
    )

    # Filter the full ranking metrics to include only selectable (non-collinear) drivers
    # This preserves detailed LLM metrics for downstream selection processes
    driver_rankings_filtered: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
        ],
    ] = {}
    # Iterate through selectable drivers to extract their corresponding metrics
    for account in selectable_drivers.keys():
        driver_rankings_filtered[account] = {}
        for class_ in selectable_drivers[account].keys():
            driver_rankings_filtered[account][class_] = {}

            # Copy metrics for each selectable driver from the full rankings
            for driver in selectable_drivers[account][class_]:
                driver_rankings_filtered[account][class_][driver] = (
                    driver_ranking_metrics[account][class_][driver]
                )

    # Return selectable drivers with their comprehensive metrics and optimal lags
    # This allows LLM to make informed driver selection decisions downstream
    return DriverAnalysisOutputLLM(
        selectable_drivers=selectable_drivers,
        metrics=driver_rankings_filtered,
        lags=best_lags,
    )


def _flatten_drivers_lags(
    lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ],
) -> dict[dts.AccountType, dict[dts.DriverName, int]]:
    """
    Flatten the nested lag dictionary structure to map each driver name directly to its optimal lag value.

    This function takes a nested dictionary of lag values organized by account, driver classification,
    and driver name, and flattens it into a simpler structure that maps each driver name directly to
    its optimal lag value for each account. This is useful for collinearity analysis where we need
    to access the optimal lag for each driver without navigating through classifications.

    Parameters
    ----------
    lags : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Nested dictionary containing optimal lag values organized by:
        - Account type (outer key)
        - Driver classification (second level key)
        - Driver name to lag value mapping (innermost dict)

    Returns
    -------
    dict[AccountType, dict[DriverName, int]]
        Flattened dictionary mapping each driver name directly to its optimal lag value for each account:
        - Account type (outer key)
        - Driver name to lag value mapping (innermost dict)
    """
    all_drivers_lags: dict[dts.AccountType, dict[dts.DriverName, int]] = {}
    for account in lags.keys():
        all_drivers_lags[account] = {}
        for classification in lags[account].keys():
            for driver in lags[account][classification].keys():
                all_drivers_lags[account][driver] = lags[account][classification][
                    driver
                ]
    return all_drivers_lags


def driver_ranking_debug(
    accounts_drivers_info: DriverAnalysisInput,
    da_params: params.DriverAnalysisParams,
) -> tuple[
    dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
        ],
    ],
    dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
        ],
    ],
    dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]],
]:
    """
    Debug driver ranking by computing full rankings and optimal lags.

    This function performs lag optimization and driver ranking, returning
    detailed ranking information for debugging and analysis purposes.

    Parameters
    ----------
    accounts_drivers_info : DriverAnalysisInput
        Input data containing account and driver information.
    da_params : params.DriverAnalysisParams
        Driver analysis parameters including lag optimization and ranking settings.

    Returns
    -------
    driver_rankings_full : dict[AccountType, dict[DriverClassification, dict[DriverName, dict[DriverRankingMetric, float]]]]
        Nested dictionary containing full driver ranking scores organized by:
        - Account type (outer key)
        - Driver classification (second level key)
        - Driver name (third level key)
        - Ranking metric to score mapping (innermost dict)
    best_lags : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Nested dictionary containing optimal lag values organized by:
        - Account type (outer key)
        - Driver classification (second level key)
        - Driver name to lag value mapping (innermost dict)

    Notes
    -----
    This function is intended for debugging and detailed analysis of the
    ranking process. It provides comprehensive information about how drivers
    are scored across different metrics.

    See Also
    --------
    rank : The underlying ranking function.
    select_best_lags : Function for lag optimization.
    driver_ranking_debug : Debug function for detailed ranking analysis.
    """
    # Step 1: Optimize lag values for each driver to maximize predictive power
    # This identifies the time offset that yields the strongest correlation
    best_lags, _ = select_best_lags(info=accounts_drivers_info, da_params=da_params)

    # Step 2: Generate comprehensive ranking scores across all metrics
    # Returns detailed scores (not just ranks) for in-depth debugging analysis
    driver_rankings_full = rank(
        accounts_drivers_info=accounts_drivers_info,
        ranking_params=da_params.ranking_params,
        best_lags=best_lags,
        max_lag=da_params.lag_params.n_max_lag,
    )

    driver_ranking_metrics = handle_ranking_metrics(
        accounts_drivers_info=accounts_drivers_info,
        ranking_params=da_params.ranking_params,
        dict_method_values=driver_rankings_full,
        best_lags=best_lags,
    )

    # Return both full ranking details and optimal lags for comprehensive debugging
    return driver_rankings_full, driver_ranking_metrics, best_lags

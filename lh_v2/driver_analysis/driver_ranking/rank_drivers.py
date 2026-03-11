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
    order_drivers_allow_ties,
    order_drivers_no_ties,
    set_lags_to_zero,
)

logger = get_logger(__name__)


def run_methods(
    info: dts.AccountDriverGroup,
    selected_methods: Sequence[rdt.DriverRankingEnum],
    ranking_params: params.RankingParams,
    logger: Optional[CustomLogger] = logger,
) -> dict[rdt.DriverRankingEnum, dict[dts.DriverName, float]]:
    """
    Execute all selected driver ranking methods and collect their results.

    Instantiates and runs each ranking method specified in ``selected_methods``,
    collecting the ranking score for each driver from every method. Each method
    produces a ``{driver_name: score}`` mapping; the results are keyed by the
    method enum so callers can inspect per-method scores.

    Execution time for each method is logged via ``logger.timing``.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        Paired account and driver time-series data for a single
        account / classification combination.
    selected_methods : Sequence[rdt.DriverRankingEnum]
        Ordered sequence of ranking methods to execute.
    ranking_params : params.RankingParams
        Configuration parameters for all ranking methods. Individual
        method params are accessed via ``ranking_params[method]``.
    logger : CustomLogger, optional
        Logger instance. Falls back to the module-level logger when
        ``None``.

    Returns
    -------
    dict[rdt.DriverRankingEnum, dict[dts.DriverName, float]]
        Mapping from each ranking method to its per-driver scores.

    Raises
    ------
    Exception
        Re-raises any exception from an individual ranking method with an
        added note identifying the failing method name.
    """

    # Fall back to the module-level logger when the caller does not supply one
    if logger is None:
        logger = get_logger(__name__)

    # Instantiate all ranking method objects up-front so that any
    # configuration or data-shape errors surface before we start timing.
    # Each method class is looked up in RANKING_METHOD_MAP by its enum key
    # and receives the shared account/driver data plus its own params slice.
    method_instances: list[AbstractRankingMethod] = [
        RANKING_METHOD_MAP[method](info, ranking_params[method])
        for method in selected_methods
    ]

    # Dict that will accumulate {method_enum: {driver_name: score}} entries
    method_values: dict[rdt.DriverRankingEnum, dict[dts.DriverName, float]] = {}

    # Initialise a wall-clock timer so we can report per-method durations
    last_time = time.time()

    # Execute each ranking method sequentially.  Methods are run in the same
    # order as `selected_methods` to guarantee deterministic output ordering.
    for method_idx in range(len(selected_methods)):
        try:
            # get_vals() returns a {DriverName: float} mapping where higher
            # values indicate a stronger relationship with the account.
            method_values[selected_methods[method_idx]] = method_instances[
                method_idx
            ].get_vals()
        except Exception as exc:
            # Annotate the exception with the failing method's human-readable
            # name so the caller can quickly identify the root cause.
            exc.add_note(
                f'Method that caused the problem - {method_instances[method_idx].name()}'
            )
            raise exc

        # Log the elapsed wall-clock time for this method and reset the timer
        logger.timing(
            f'Time of method `{method_instances[method_idx].name()}`'
            f' - {time.time() - last_time:.4f} seconds.'
        )
        last_time = time.time()

    return method_values


def rank(
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
        dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
    ],
]:
    """
    Rank drivers across multiple statistical methods for every account/classification pair.

    For each ``(account_type, driver_classification)`` combination the function:

    1. Applies lag-adjusted time-series data.
    2. Runs every selected ranking method via :func:`run_methods`.
    3. Converts per-method scores into per-method ranks (ties allowed).
    4. Averages the per-method ranks and produces a single tie-free final rank.
    5. Stores the final rank, average rank, and every raw method score in the
       returned nested dictionary.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Container holding all account groups and their classified driver groups.
    ranking_params : params.RankingParams
        Configuration specifying which ranking methods to use, their
        hyper-parameters, and the list of output ranking metrics.
    best_lags : dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]], optional
        Optimal lag (in periods) for each driver, keyed by account type and
        classification. When ``None``, all lags default to zero.
    max_lag : int, default 0
        Global maximum lag used to align (truncate) the front of all
        time-series so they share a common start index.
    logger : CustomLogger, optional
        Logger instance. Falls back to the module-level logger when ``None``.

    Returns
    -------
    dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]]]]
        Nested mapping ``{account_type: {classification: {driver: {metric: value}}}}``
        containing ``FINAL_RANK``, ``AVG_RANK``, and every individual method
        score for each driver.
    """
    # Fall back to the module-level logger when the caller does not supply one
    if logger is None:
        logger = get_logger(__name__)
    logger.info('Ranking drivers.')

    # Resolve the set of ranking methods enabled in the configuration.
    # `select_methods` reads boolean flags from `ranking_params.methods`
    # and returns only those whose flag is True.
    selected_methods = select_methods(methods_params=ranking_params.methods)

    # Guard: at least one method must be active, otherwise the downstream
    # averaging / rank-breaking logic would produce empty results.
    assert len(selected_methods) > 0, (
        'Must use at least one Ranking method, check config file.'
    )

    # When the caller does not supply per-driver lag information (e.g. on
    # the first pass before lag optimisation), default every driver to a
    # lag of zero so the rest of the pipeline can proceed uniformly.
    if best_lags is None:
        best_lags = set_lags_to_zero(accounts_drivers_info=accounts_drivers_info)

    # Top-level accumulator: account → classification → driver → metric → value.
    # Populated incrementally inside the nested loops below.
    ranking_values: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
        ],
    ] = {}

    # --- Outer loop: iterate over every account type (e.g. Revenue, COGS) ---
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        ranking_values[account] = {}

        # --- Inner loop: iterate over driver classifications
        #     (e.g. positive, negative, neutral) within this account ---
        for (
            class_
        ) in accounts_drivers_info.classified_drivers.get_ordered_classifications():
            ranking_values[account][class_] = {}

            # Build an AccountDriverGroup that pairs the single account
            # time-series with the classified driver group, after applying
            # each driver's optimal lag.  `apply_lag` on the account trims
            # the front by `max_lag` periods so all series share the same
            # start index.  `apply_lags` shifts each driver individually
            # according to `best_lags` and also trims to `max_lag`.
            max_lag = max(best_lags[account][class_].values())
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

            # Execute every selected ranking method on this (account, class)
            # pair and collect a {method: {driver: score}} mapping.
            method_values = run_methods(
                info=info,
                selected_methods=selected_methods,
                ranking_params=ranking_params,
                logger=logger,
            )

            # ----------------------------------------------------------
            # Stage 1: Assemble a (num_methods × num_drivers) score matrix.
            # Rows = methods (in selection order), columns = drivers (in
            # their canonical order from `get_ordered_drivers`).
            # ----------------------------------------------------------
            arr_method_vals: ArrayF = np.zeros(
                (len(selected_methods), len(info.drivers)), dtype=np.float32
            )
            for method_idx, method in enumerate(selected_methods):
                for driver_idx, driver in enumerate(info.drivers.get_ordered_drivers()):
                    arr_method_vals[method_idx, driver_idx] = method_values[method][
                        driver
                    ]

            # ----------------------------------------------------------
            # Stage 2: Convert each method's raw scores into ranks.
            # Ties ARE allowed here — two drivers with identical scores
            # from the same method receive the same rank.
            # ----------------------------------------------------------
            arr_method_ranks: ArrayI = np.zeros_like(arr_method_vals, dtype=np.int32)
            for method_idx, method in enumerate(selected_methods):
                arr_method_ranks[method_idx] = order_drivers_allow_ties(
                    values=arr_method_vals[method_idx]
                )

            # ----------------------------------------------------------
            # Stage 3: Compute a consensus ranking by averaging each
            # driver's rank across all methods, then producing a final
            # unique (tie-free) ordering from those averages.
            # ----------------------------------------------------------
            arr_avg_ranks: ArrayF = np.mean(arr_method_ranks, axis=0)
            arr_final_ranks: ArrayI = order_drivers_no_ties(
                arr_avg_rankings=arr_avg_ranks
            )

            # ----------------------------------------------------------
            # Stage 4: Pack results into the output dictionary.
            # ----------------------------------------------------------

            # Pre-create an empty metrics dict for each driver so that
            # key insertion order matches the canonical driver ordering.
            for driver in info.drivers.get_ordered_drivers():
                ranking_values[account][class_][driver] = {}

            # Store the two aggregate metrics (FINAL_RANK, AVG_RANK).
            # Values are cast to float for JSON-serialisability.
            for idx, driver in enumerate(info.drivers.get_ordered_drivers()):
                ranking_values[account][class_][driver][
                    rdt.DriverRankingMetric.FINAL_RANK
                ] = float(arr_final_ranks[idx])
                ranking_values[account][class_][driver][
                    rdt.DriverRankingMetric.AVG_RANK
                ] = float(arr_avg_ranks[idx])

            # Append each individual method's raw score under its
            # corresponding DriverRankingMetric enum value so callers can
            # inspect or export per-method detail alongside the aggregates.
            for method in method_values.keys():
                for driver in info.drivers.get_ordered_drivers():
                    ranking_values[account][class_][driver][
                        rdt.DriverRankingMetric(method.value)
                    ] = method_values[method][driver]

    return ranking_values


def handle_ranking_metrics(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    ranking_params: params.RankingParams,
    dict_method_values: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
        ],
    ],
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ],
) -> dict[
    dts.AccountType,
    dict[
        dts.DriverClassification,
        dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
    ],
]:
    """
    Collect the user-requested ranking metrics for every driver.

    Iterates over each ``(account_type, classification)`` pair and assembles
    only the metrics listed in ``ranking_params.ranking_metrics``.  Metrics
    that were already computed during :func:`rank` (present in
    ``dict_method_values``) are copied directly; any additional metrics are
    computed on-the-fly by instantiating and running the corresponding
    ranking method.

    Drivers are ordered by ``FINAL_RANK`` so the output dict preserves
    insertion-order from best to worst.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Container holding all account groups and their classified driver
        groups.
    ranking_params : params.RankingParams
        Configuration specifying which ranking metrics to include in the
        output.
    dict_method_values : dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]]]]
        Pre-computed ranking results returned by :func:`rank`.
    best_lags : dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]]
        Optimal lag for each driver, used to reconstruct lag-adjusted data
        when a metric must be computed on-the-fly.

    Returns
    -------
    dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]]]]
        Subset (or superset) of ``dict_method_values`` containing exactly
        the metrics requested in ``ranking_params.ranking_metrics``, with
        drivers ordered by final rank.
    """
    ranking_metrics: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
        ],
    ] = {}

    # Build the set of method-level metrics that were already evaluated
    # during the main `rank` pass.  This lets us avoid redundant computation
    # for any metric the user requests that was part of the original run.
    ran_metrics_set = _get_ran_metrics_set(dict_method_values=dict_method_values)

    # --- Outer loop: iterate over every account type ---
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        ranking_metrics[account] = {}

        # --- Inner loop: iterate over driver classifications ---
        for (
            class_
        ) in accounts_drivers_info.classified_drivers.get_ordered_classifications():
            ranking_metrics[account][class_] = {}

            # Derive max_lag from the best lags for this (account, class) pair.
            max_lag = max(best_lags[account][class_].values())

            # Reconstruct the lag-adjusted account + driver bundle.  This
            # is only needed when on-the-fly metric computation is required,
            # but we build it unconditionally to keep the code simple.
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

            # Order drivers by their previously computed FINAL_RANK so that
            # the output dict's insertion order goes best → worst.  This is
            # useful for consumers that iterate dicts in insertion order.
            drivers_ordered = _order_drivers_by_metric(
                dict_method_values=dict_method_values[account][class_],
                metric=rdt.DriverRankingMetric.FINAL_RANK,
            )

            # Pre-create an empty metrics dict for each driver in rank order
            for driver in drivers_ordered:
                ranking_metrics[account][class_][driver] = {}

            # Process each user-requested metric.  We use a match/case to
            # distinguish between the two aggregate metrics (always cached)
            # and method-level metrics (may or may not be cached).
            for ranking_metric in ranking_params.ranking_metrics:
                match ranking_metric:
                    # FINAL_RANK and AVG_RANK are always present in
                    # dict_method_values because they are computed during
                    # the aggregation stage of `rank()`.  Copy directly.
                    case (
                        rdt.DriverRankingMetric.FINAL_RANK
                        | rdt.DriverRankingMetric.AVG_RANK
                    ):
                        for driver in drivers_ordered:
                            ranking_metrics[account][class_][driver][ranking_metric] = (
                                dict_method_values[account][class_][driver][
                                    ranking_metric
                                ]
                            )

                    # For all other metrics, check whether the value was
                    # already computed.  If yes, reuse it; otherwise
                    # instantiate the ranking method and compute on-the-fly.
                    case _:
                        if ranking_metric in ran_metrics_set:
                            # Metric was computed in the main `rank` pass —
                            # copy the cached value to avoid redundant work.
                            for driver in drivers_ordered:
                                ranking_metrics[account][class_][driver][
                                    ranking_metric
                                ] = dict_method_values[account][class_][driver][
                                    ranking_metric
                                ]
                        else:
                            # Metric was NOT part of the original ranking
                            # methods, so it needs to be computed now.
                            # Convert the metric enum back to its method
                            # enum counterpart and instantiate the method.
                            method_enum = rdt.DriverRankingEnum(ranking_metric.value)
                            method_instance = RANKING_METHOD_MAP[method_enum](
                                account_driver_info=info,
                                method_params=ranking_params[method_enum],
                            )

                            try:
                                # Run the method to obtain per-driver scores
                                method_values = method_instance.get_vals()
                            except Exception as exc:
                                # Tag the exception with the method name for
                                # easier debugging upstream
                                exc.add_note(
                                    f'Method that caused the problem - {method_instance.name()}'
                                )
                                raise exc

                            # Store the freshly computed scores for every
                            # driver under this metric key
                            for driver in drivers_ordered:
                                ranking_metrics[account][class_][driver][
                                    ranking_metric
                                ] = method_values[driver]

    return ranking_metrics


def _get_ran_metrics_set(
    dict_method_values: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
        ],
    ],
) -> set[rdt.DriverRankingMetric]:
    """
    Identify which method-level metrics have already been computed.

    Peeks at the first driver entry in ``dict_method_values`` to discover
    the set of stored metrics, then removes the aggregate metrics
    (``FINAL_RANK`` and ``AVG_RANK``) that are not method-level scores.

    Parameters
    ----------
    dict_method_values : dict[
        dts.AccountType, dict[
            dts.DriverClassification, dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]]
        ]
    ]
        Full ranking results as returned by :func:`rank`.

    Returns
    -------
    set[rdt.DriverRankingMetric]
        Method-level metrics already present in ``dict_method_values``.
    """
    # Navigate into the nested dict three levels deep to reach an arbitrary
    # driver's metrics dict.  All drivers share the same set of metric keys,
    # so inspecting any single driver is sufficient.
    acc0 = next(iter(dict_method_values))  # first account type
    class0 = next(iter(dict_method_values[acc0]))  # first classification
    driver0 = next(iter(dict_method_values[acc0][class0]))  # first driver

    # Copy the full set of metric keys stored for this driver
    ran_metrics_set = set(dict_method_values[acc0][class0][driver0].keys())

    # Remove aggregate metrics (AVG_RANK, FINAL_RANK) — those are always
    # present but are computed separately from individual method scores and
    # should not be matched against method-level metric requests.
    ran_metrics_set.discard(rdt.DriverRankingMetric.AVG_RANK)
    ran_metrics_set.discard(rdt.DriverRankingMetric.FINAL_RANK)
    return ran_metrics_set


def _order_drivers_by_metric(
    dict_method_values: dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]],
    metric: rdt.DriverRankingMetric = rdt.DriverRankingMetric.FINAL_RANK,
) -> list[dts.DriverName]:
    """
    Sort driver names in ascending order of a given ranking metric.

    Parameters
    ----------
    dict_method_values : dict[dts.DriverName, dict[rdt.DriverRankingMetric, float]]
        Per-driver metric scores for a single account/classification pair.
    metric : rdt.DriverRankingMetric, default ``FINAL_RANK``
        The metric whose value is used as the sort key.

    Returns
    -------
    list[dts.DriverName]
        Driver names sorted from best (lowest value) to worst.
    """
    return sorted(
        dict_method_values.keys(),
        key=lambda driver: dict_method_values[driver][metric],
    )

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from .collinearity_util import use_driver_pruning

logger = get_logger(__name__)


def prune_final_drivers(
    drivers_map: dict[dts.DriverName, int],
    classified_driver_rankings: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ],
    da_params: params.DriverAnalysisParams,
    preselected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ],
    arr_collinearity_dict: dict[dts.AccountType, ArrayF],
) -> dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]]:
    """Prune collinear drivers across all accounts for LLM-based selection.

    Iterates over every account and delegates to
    :func:`prune_final_drivers_account` to remove highly collinear
    candidates while preserving ranking order.  The pruning uses a more
    permissive threshold than the strict collinearity removal applied
    earlier in the pipeline, so that the downstream LLM still has
    meaningful choice among diverse drivers.

    Parameters
    ----------
    drivers_map : dict[dts.DriverName, int]
        Global mapping of driver names to their column indices in the
        collinearity matrix.
    classified_driver_rankings : dict
        Nested mapping ``{account -> {classification -> {driver -> rank}}}``
        where lower rank values indicate higher-priority drivers.
    da_params : params.DriverAnalysisParams
        Driver-analysis parameters, including collinearity thresholds and
        the desired number of final drivers per classification.
    preselected_drivers : dict
        Nested mapping ``{account -> {classification -> [driver, ...]}}``
        of drivers that must always be included regardless of
        collinearity.
    arr_collinearity_dict : dict[dts.AccountType, ArrayF]
        Per-account collinearity matrices of shape
        ``(n_drivers, n_drivers)``.

    Returns
    -------
    dict[AccountType, dict[DriverClassification, list[DriverName]]]
        The pruned set of selectable drivers for each account and
        classification, ready for LLM consideration.

    See Also
    --------
    prune_final_drivers_account : Per-account pruning logic.
    """
    final_pruned_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}

    for account in classified_driver_rankings.keys():
        final_pruned_drivers[account] = prune_final_drivers_account(
            drivers_map=drivers_map,
            classified_driver_rankings=classified_driver_rankings[account],
            da_params=da_params,
            preselected_drivers=preselected_drivers[account],
            arr_collinearity=arr_collinearity_dict[account],
        )

    return final_pruned_drivers


def prune_final_drivers_account(
    drivers_map: dict[dts.DriverName, int],
    classified_driver_rankings: dict[
        dts.DriverClassification, dict[dts.DriverName, int]
    ],
    da_params: params.DriverAnalysisParams,
    preselected_drivers: dict[dts.DriverClassification, list[dts.DriverName]],
    arr_collinearity: ArrayF,
) -> dict[dts.DriverClassification, list[dts.DriverName]]:
    """
    Prune highly collinear drivers while preserving ranking order for LLM selection.

    This helper function filters drivers by removing those that are highly collinear
    with already-allowed drivers, using a more permissive threshold than strict
    collinearity removal. The result is a pruned set of selectable drivers for
    downstream LLM-based selection.

    Parameters
    ----------
    drivers_map : dict[dts.DriverName, int]
        Global mapping of driver names to their column indices in the
        collinearity matrix.
    classified_driver_rankings : dict[DriverClassification, dict[DriverName, int]]
        Rankings of drivers within each classification for a single
        account, where lower rank values indicate higher priority.
    da_params : params.DriverAnalysisParams
        Driver-analysis parameters.  The key fields used here are
        ``collinearity_params.threashold_base_pruning`` (the permissive
        collinearity threshold) and ``n_final_drivers_per_classification``
        (the minimum number of drivers that should survive pruning).
    preselected_drivers : dict[DriverClassification, list[DriverName]]
        Drivers that are pre-selected (e.g. by business rules) and must
        be included in the allowed set for each classification.  These
        seed the allowed list before rank-ordered pruning begins.
    arr_collinearity : ArrayF
        Collinearity matrix of shape ``(n_drivers, n_drivers)`` for the
        current account, where element ``[i, j]`` is the pairwise
        collinearity score between drivers *i* and *j*.

    Returns
    -------
    dict[DriverClassification, list[DriverName]]
        Mapping of each driver classification to its list of allowed
        (non-highly-collinear) driver names, in rank order.

    Notes
    -----
    * Uses :func:`use_driver_pruning` which applies a logarithmically
      decaying threshold — more permissive than the strict
      :func:`use_driver` used elsewhere — so the LLM receives a wider
      candidate pool.
    * Pruning is performed *within* each classification independently;
      collinearity is only checked against drivers already allowed in
      the **same** classification.
    * If a classification ends up with fewer drivers than
      ``n_final_drivers_per_classification + 1``, the fallback helper
      :func:`_ensure_correct_num_drivers_allowed` progressively relaxes
      the threshold to admit more drivers.

    See Also
    --------
    use_driver_pruning : Evaluates a single driver against the pruning threshold.
    _ensure_correct_num_drivers_allowed : Fallback that relaxes the threshold.
    prune_final_drivers : Outer function that calls this helper per account.
    """
    # ── Step 1: Invert rankings so we can look up driver names by rank ──
    # Original: {driver_name: rank}  →  Flipped: {rank: driver_name}
    flipped_classified_driver_rankings: dict[
        dts.DriverClassification, dict[int, dts.DriverName]
    ] = {
        classification: {
            val: key for key, val in classified_driver_rankings[classification].items()
        }
        for classification in classified_driver_rankings.keys()
    }

    # ── Step 2: Build rank-ordered driver lists per classification ──
    # Produces a list where index 0 is the highest-ranked driver (rank 1).
    classified_orderings: dict[dts.DriverClassification, list[dts.DriverName]] = {}

    for classification in classified_driver_rankings.keys():
        classified_orderings[classification] = []
        for k in range(len(classified_driver_rankings[classification].keys())):
            # Ranks are 1-indexed, so look up rank k+1
            classified_orderings[classification].append(
                flipped_classified_driver_rankings[classification][k + 1]
            )

    # ── Step 3: Seed the allowed set with preselected (must-include) drivers ──
    # NOTE: this aliases the incoming dict; mutations below therefore accumulate
    # into the same lists that were passed in.
    allowed_drivers: dict[dts.DriverClassification, list[dts.DriverName]] = (
        preselected_drivers
    )

    # ── Step 4: Greedily add drivers in rank order, skipping collinear ones ──
    for classification in classified_driver_rankings.keys():
        for driver in classified_orderings[classification]:
            if len(allowed_drivers[classification]) > 0:
                # Check pairwise collinearity against every already-allowed driver
                if use_driver_pruning(
                    arr_collinearity=arr_collinearity,
                    driver_ind=drivers_map[driver],
                    used_driver_inds=[
                        drivers_map[_driver]
                        for _driver in allowed_drivers[classification]
                    ],
                    threashold_base=da_params.collinearity_params.threashold_base_pruning,
                ):
                    allowed_drivers[classification].append(driver)
            else:
                # No drivers yet — accept the highest-ranked one unconditionally
                allowed_drivers[classification].append(driver)

        # ── Step 5: Fallback — relax the threshold if too few drivers survived ──
        # The "+1" accounts for the preselected driver that seeds the list.
        if (
            len(allowed_drivers[classification])
            < da_params.n_final_drivers_per_classification + 1
        ):
            logger.warning(
                f'After pruning, classification {classification} has only '
                f'{len(allowed_drivers[classification])} drivers, which is less than the desired '
                f'{da_params.n_final_drivers_per_classification}. This may indicate that many drivers are collinear.'
            )
            allowed_drivers[classification] = _ensure_correct_num_drivers_allowed(
                drivers_map=drivers_map,
                allowed_drivers_class=allowed_drivers[classification],
                class_drivers=classified_orderings[classification],
                da_params=da_params,
                arr_collinearity=arr_collinearity,
            )

    return allowed_drivers


def _ensure_correct_num_drivers_allowed(
    drivers_map: dict[dts.DriverName, int],
    allowed_drivers_class: list[dts.DriverName],
    class_drivers: list[dts.DriverName],
    da_params: params.DriverAnalysisParams,
    arr_collinearity: ArrayF,
) -> list[dts.DriverName]:
    """Progressively relax the collinearity threshold to reach a minimum driver count.

    When the initial greedy pruning pass in :func:`prune_final_drivers_account`
    leaves a classification with fewer drivers than required, this fallback
    iteratively raises the effective threshold until enough drivers are admitted
    or a maximum number of iterations is reached.

    On each iteration the effective threshold is:

    .. math::

        T_{\\text{eff}} = T_{\\text{base}} \\times 1.1^{(c+1)}

    where *c* is the zero-based iteration counter.  This exponential relaxation
    allows increasingly collinear drivers to pass, prioritising diversity of
    choice for the downstream LLM.

    Parameters
    ----------
    drivers_map : dict[dts.DriverName, int]
        Global mapping of driver names to collinearity-matrix indices.
    allowed_drivers_class : list[dts.DriverName]
        Drivers already admitted for this classification (mutated in-place
        and also returned).
    class_drivers : list[dts.DriverName]
        All candidate drivers for this classification, in rank order.
    da_params : params.DriverAnalysisParams
        Parameters providing ``collinearity_params.threashold_base_pruning``
        and ``n_final_drivers_per_classification``.
    arr_collinearity : ArrayF
        Collinearity matrix of shape ``(n_drivers, n_drivers)``.

    Returns
    -------
    list[dts.DriverName]
        The (potentially enlarged) list of allowed drivers for this
        classification.
    """
    threashold_base = da_params.collinearity_params.threashold_base_pruning
    max_iter = 10  # cap iterations to avoid runaway relaxation
    multiplier = 1.1  # geometric factor applied to the threshold each round
    c = 0

    while (
        len(allowed_drivers_class) < da_params.n_final_drivers_per_classification + 1
        and c < max_iter
    ):
        # Re-scan candidates that were rejected in previous passes, using a
        # progressively more permissive threshold.
        for driver in class_drivers:
            if driver not in allowed_drivers_class:
                if use_driver_pruning(
                    arr_collinearity=arr_collinearity,
                    driver_ind=drivers_map[driver],
                    used_driver_inds=[
                        drivers_map[_driver] for _driver in allowed_drivers_class
                    ],
                    threashold_base=threashold_base * multiplier ** (c + 1),
                ):
                    allowed_drivers_class.append(driver)
        c += 1

    return allowed_drivers_class

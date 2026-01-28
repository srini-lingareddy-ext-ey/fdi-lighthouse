import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from .collinearity_util import run_methods, select_methods, use_driver_pruning

logger = get_logger(__name__)


def _prune_drivers(
    drivers_info: dts.DriverGroup,
    classified_driver_rankings: dict[
        dts.DriverClassification, dict[dts.DriverName, int]
    ],
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
    drivers_info : dts.DriverGroup
        Group containing driver data and metadata, including driver name mapping.
    classified_driver_rankings : dict[DriverClassification, dict[DriverName, int]]
        Rankings of drivers within each classification, where lower ranks indicate
        higher priority.
    preselected_drivers : dict[DriverClassification, list[DriverName]]
        Drivers that are preselected and must be included in the allowed set for
        each classification.
    arr_collinearity : ArrayF
        Collinearity matrix of shape (n_drivers, n_drivers) containing collinearity
        scores between driver pairs.

    Returns
    -------
    dict[DriverClassification, list[DriverName]]
        Dictionary mapping each driver classification to the list of allowed
        (non-highly-collinear) driver names.

    Notes
    -----
    This function uses `use_driver_pruning` which applies a less strict threshold
    than `use_driver`, allowing more drivers to pass through for LLM consideration.
    The pruning is done within each classification independently, considering only
    collinearity with drivers already allowed in that same classification.

    See Also
    --------
    use_driver_pruning : Function that determines if a driver passes the pruning threshold.
    prune_collinearity : Main function that calls this helper for each account.
    """
    # Step 1: Invert the ranking dictionary for each classification
    # This allows lookup of driver name by rank (rank -> driver name)
    flipped_classified_driver_rankings: dict[
        dts.DriverClassification, dict[int, dts.DriverName]
    ] = {
        classification: {
            val: key for key, val in classified_driver_rankings[classification].items()
        }
        for classification in classified_driver_rankings.keys()
    }

    # Step 2: Build ordered queues of drivers by their rankings for each classification
    classified_orderings: dict[dts.DriverClassification, list[dts.DriverName]] = {}

    for classification in classified_driver_rankings.keys():
        classified_orderings[classification] = []
        # Add drivers to the queue in order of their rank (lowest rank first)
        for k in range(len(classified_driver_rankings[classification].keys())):
            classified_orderings[classification].append(
                flipped_classified_driver_rankings[classification][k + 1]
            )

    # Step 3: Initialize dict of lists to store selected drivers, starting with preselected drivers
    allowed_drivers: dict[dts.DriverClassification, list[dts.DriverName]] = (
        preselected_drivers
    )

    # Step 4: Select allowed drivers for each classification
    for classification in classified_driver_rankings.keys():
        for driver in classified_orderings[classification]:
            if len(allowed_drivers[classification]) > 0:
                # Only add driver if not collinear with already selected drivers
                if use_driver_pruning(
                    arr_collinearity=arr_collinearity,
                    driver_ind=drivers_info.map[driver],
                    used_driver_inds=[
                        drivers_info.map[_driver]
                        for _driver in allowed_drivers[classification]
                    ],
                ):
                    allowed_drivers[classification].append(driver)
            else:
                # If no drivers selected yet, add the first driver unconditionally
                allowed_drivers[classification].append(driver)

    return allowed_drivers


def prune_collinearity(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    classified_driver_rankings: dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[dts.DriverName, int]],
    ],
    da_params: params.DriverAnalysisParams,
    best_lags: dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[dts.DriverName, int]],
    ],
) -> dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]]:
    """
    Prune collinear drivers for LLM-based driver selection.

    This function performs collinearity analysis and filters out highly collinear
    drivers, but uses a more permissive threshold than `remove_collinearity`. The
    result is a larger set of selectable drivers that the LLM can choose from,
    while still avoiding extreme multicollinearity.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Container with account groups and their associated classified driver groups.
    classified_driver_rankings : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Nested dictionary containing driver rankings for each account type and
        classification. Lower rank numbers indicate higher priority.
    da_params : params.DriverAnalysisParams
        Driver analysis parameters including collinearity method settings.
    best_lags : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Optimal lag values for each driver, organized by account type and
        classification. Used to apply appropriate temporal offsets before
        collinearity analysis.

    Returns
    -------
    dict[AccountType, dict[DriverClassification, list[DriverName]]]
        Pruned set of drivers for each account type and classification. This is
        the pool of drivers available for LLM-based selection.

    Notes
    -----
    The function performs the following steps:
    1. Selects collinearity detection methods based on configuration
    2. Combines all classified drivers into a single group for analysis
    3. Applies optimal lags to driver data for each account
    4. Computes collinearity matrices using selected methods
    5. Prunes drivers using a permissive threshold to create LLM selection pool

    Unlike `remove_collinearity`, this function does not limit the number of
    drivers returned. Instead, it filters out only the most highly collinear
    drivers, preserving a larger pool for the LLM to make final selections from.

    See Also
    --------
    remove_collinearity : Stricter collinearity removal for direct driver selection.
    _prune_drivers : Helper function that performs pruning for one account.
    use_driver_pruning : Threshold function used for pruning decisions.

    Examples
    --------
    >>> pruned = prune_collinearity(
    ...     accounts_drivers_info=data,
    ...     classified_driver_rankings=rankings,
    ...     da_params=params,
    ...     best_lags=lags
    ... )
    >>> revenue_pruned = pruned[AccountType('REVENUE')]
    >>> primary_selectable = revenue_pruned[DriverClassification.PRIMARY]
    >>> # LLM can now select from primary_selectable drivers
    """
    # Select which collinearity detection methods to use based on configuration
    selected_methods = select_methods(collinearity_params=da_params.collinearity_params)

    # Combine all classified drivers into a single unified DriverGroup
    # This enables collinearity analysis across all driver classifications
    drivers_info = accounts_drivers_info.classified_drivers.combine()

    # Initialize dictionary to store collinearity matrices for each account
    arr_collinearity_dict: dict[dts.AccountType, ArrayF] = {}

    # Compute collinearity matrix for each account independently
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        # Flatten lag dictionary structure: combine all classifications into single dict
        # This maps each driver name directly to its optimal lag value
        all_drivers_lags: dict[dts.DriverName, int] = {}
        for (
            classification
        ) in accounts_drivers_info.classified_drivers.classification_groups.keys():
            # Get drivers for this classification and extract their lag values
            for driver in accounts_drivers_info.classified_drivers.maps[
                accounts_drivers_info.classified_drivers.classification_groups[
                    classification
                ]
            ].keys():
                # Store the optimal lag for this driver from this account's lag dictionary
                all_drivers_lags[driver] = best_lags[account][classification][driver]

        # Find the maximum lag across all drivers to determine truncation amount
        max_lag = max(all_drivers_lags.values())

        # Apply lags to driver data and compute collinearity matrix
        # Lagging ensures temporal alignment before measuring collinearity
        arr_collinearity_dict[account] = run_methods(
            drivers_info=drivers_info.apply_lags(
                best_lags=all_drivers_lags, max_lag=max_lag
            ),
            selected_methods=selected_methods,
            collinearity_params=da_params.collinearity_params,
        )

    # Initialize preselected drivers structure (currently empty)
    # Preselected drivers would be those that must be included regardless of collinearity
    # TODO: Add configuration support for preselected drivers
    preselected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        # Create empty list for each classification - will be populated from config in future
        preselected_drivers[account] = {
            classification: []
            for classification in accounts_drivers_info.classified_drivers.get_ordered_classifications()
        }

    # Initialize dictionary to store final pruned drivers for each account and classification
    final_pruned_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}

    # Prune drivers for each account independently using permissive threshold
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        # Call helper function to perform pruning for this account
        # Uses use_driver_pruning which is more lenient than use_driver
        final_pruned_drivers[account] = _prune_drivers(
            drivers_info=drivers_info,
            arr_collinearity=arr_collinearity_dict[account],
            classified_driver_rankings=classified_driver_rankings[account],
            preselected_drivers=preselected_drivers[account],
        )

    # Return the pruned set of selectable drivers for LLM-based selection
    return final_pruned_drivers

from collections import deque

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from .collinearity_util import run_methods, select_methods, use_driver

logger = get_logger(__name__)


def _select_final_drivers(
    drivers_info: dts.DriverGroup,
    classified_driver_rankings: dict[
        dts.DriverClassification, dict[dts.DriverName, int]
    ],
    da_params: params.DriverAnalysisParams,
    preselected_drivers: dict[dts.DriverClassification, list[dts.DriverName]],
    arr_collinearity: ArrayF,
) -> dict[dts.DriverClassification, list[dts.DriverName]]:
    """
    Selects the final set of drivers for each classification, removing collinear drivers
    from the ranked candidates and ensuring preselected drivers are included.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        Group containing driver data and metadata, including driver name mapping.
    classified_driver_rankings : dict of {dts.DriverClassification : dict of {dts.DriverName : int}}
        Rankings of drivers within each classification, where a lower rank indicates higher priority.
    general_params : params.GeneralParams
        General configuration parameters, including the number of final drivers to select per classification.
    preselected_drivers : dict of {dts.DriverClassification : list of dts.DriverName}
        Drivers that are preselected and must be included in the final set for each classification.
    arr_collinearity : ArrayF
        Collinearity matrix of shape (n_drivers, n_drivers) containing collinearity scores between driver pairs.

    Returns
    -------
    dict[dts.DriverClassification, list[dts.DriverName]]
        Dictionary mapping each driver classification to the final selected list of driver names.

    Warns
    -----
    UserWarning
        If the maximum number of iterations is reached during driver selection, indicating
        potential issues with the selection process.

    Notes
    -----
    The function performs the following steps:
        1. Inverts the ranking dictionary to enable lookup by rank.
        2. Creates ordered queues of drivers sorted by ranking.
        3. Iteratively selects drivers that are not collinear with already selected drivers.
        4. Continues until the desired number of drivers per classification is reached
           or no more candidates are available.
        5. Preselected drivers are always included in the final selection.
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
    classified_orderings: dict[dts.DriverClassification, deque[dts.DriverName]] = {}

    for classification in classified_driver_rankings.keys():
        classified_orderings[classification] = deque()
        # Add drivers to the queue in order of their rank (lowest rank first)
        for k in range(len(classified_driver_rankings[classification].keys())):
            classified_orderings[classification].append(
                flipped_classified_driver_rankings[classification][k + 1]
            )

    # Step 3: Initialize dict of lists to store selected drivers, starting with preselected drivers
    classified_selected_drivers: dict[
        dts.DriverClassification, list[dts.DriverName]
    ] = preselected_drivers
    # Flatten all selected drivers into a single list for collinearity checks
    _selected_drivers: list[dts.DriverName] = []
    for classification in classified_selected_drivers.keys():
        _selected_drivers += classified_selected_drivers[classification]

    # Step 4: Set maximum iterations to avoid infinite loops during selection
    max_iter = 2 * arr_collinearity.shape[0]

    # Step 5: Select drivers for each classification
    for classification in classified_driver_rankings.keys():
        c = 0  # iteration counter
        while (
            len(classified_selected_drivers[classification])
            < da_params.n_final_drivers_per_classification
            and len(classified_orderings[classification]) > 0
            and c < max_iter
        ):
            # Pop the next highest ranked driver from the queue
            driver = classified_orderings[classification].popleft()
            if len(_selected_drivers) > 0:
                # Only add driver if not collinear with already selected drivers
                if use_driver(
                    arr_collinearity=arr_collinearity,
                    driver_ind=drivers_info.map[driver],
                    used_driver_inds=[
                        drivers_info.map[_driver] for _driver in _selected_drivers
                    ],
                ):
                    classified_selected_drivers[classification].append(driver)
                    _selected_drivers.append(driver)
            else:
                # If no drivers selected yet, add the first driver unconditionally
                classified_selected_drivers[classification].append(driver)
                _selected_drivers.append(driver)
            c += 1

        logger.debug(
            f'Selected Drivers for Classification `{classification}`: '
            f'`{classified_selected_drivers[classification]}`'
        )

        # Warn if maximum iterations were reached (should not happen in normal operation)
        if c == max_iter:
            logger.warning(
                f'Max iteration reached when selecting final set of '
                f'drivers for Classification {classification}, '
                f'this behavior is not expected, and should be investigated.'
            )

    # Return the final selected drivers for each classification
    return classified_selected_drivers


def remove_collinearity(
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
    Remove collinear drivers and select final driver set for each account and classification.

    This function performs collinearity analysis across all drivers for each account type,
    then selects a final set of non-collinear drivers based on their rankings. The selection
    process ensures that chosen drivers are sufficiently independent to avoid multicollinearity
    issues in downstream modeling.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Container with account groups and their associated classified driver groups.
    classified_driver_rankings : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Nested dictionary containing driver rankings for each account type and classification.
        Lower rank numbers indicate higher priority for selection.
    da_params : params.DriverAnalysisParams
        Driver analysis parameters including collinearity method settings and the number
        of final drivers to select per classification.
    best_lags : dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        Optimal lag values for each driver, organized by account type and classification.
        Used to apply appropriate temporal offsets before collinearity analysis.

    Returns
    -------
    dict[AccountType, dict[DriverClassification, list[DriverName]]]
        Final selected drivers for each account type and classification. Drivers are
        selected to minimize collinearity while respecting ranking priorities.

    Notes
    -----
    The function performs the following steps:
    1. Selects collinearity detection methods based on configuration
    2. Combines all classified drivers into a single group for analysis
    3. Applies optimal lags to driver data for each account
    4. Computes collinearity matrices using selected methods
    5. Iteratively selects top-ranked drivers that are not collinear with existing selections

    Currently, preselected drivers (drivers that must be included) are initialized as empty
    lists. Future enhancement will add configuration support for preselected drivers.

    See Also
    --------
    _select_final_drivers : Helper function that performs driver selection for one account.
    run_methods : Executes collinearity detection methods.
    select_methods : Determines which collinearity methods to use.

    Examples
    --------
    >>> selected = remove_collinearity(
    ...     accounts_drivers_info=data,
    ...     classified_driver_rankings=rankings,
    ...     da_params=params,
    ...     best_lags=lags
    ... )
    >>> revenue_drivers = selected[AccountType('REVENUE')]
    >>> primary_drivers = revenue_drivers[DriverClassification.PRIMARY]
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
            # Get the index for this classification to access its driver map
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

    # TODO: Need to add preselected driver support in the config/params
    # Add a preselected drivers key in the config file, and
    # a corresponding key in the params, will need to be on a per
    # driver classification level, and will need to verify correct
    # driver classifications

    # Initialize preselected drivers structure (currently empty, to be populated from config)
    # Preselected drivers are those that must be included regardless of collinearity
    preselected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}
    for account in accounts_drivers_info.accounts.account_map.keys():
        # Create empty list for each classification - will be populated from config in future
        preselected_drivers[account] = {
            classification: []
            for classification in accounts_drivers_info.classified_drivers.get_ordered_classifications()
        }

    # Initialize dictionary to store final selected drivers for each account and classification
    final_selected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}

    # Select final non-collinear drivers for each account independently
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        # Call helper function to perform iterative selection for this account
        # Selection respects rankings while avoiding collinear drivers
        final_selected_drivers[account] = _select_final_drivers(
            drivers_info=drivers_info,
            classified_driver_rankings=classified_driver_rankings[account],
            da_params=da_params,
            preselected_drivers=preselected_drivers[account],
            arr_collinearity=arr_collinearity_dict[account],
        )

    # Return the final selected drivers organized by account and classification
    return final_selected_drivers

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
    selected_methods = select_methods(collinearity_params=da_params.collinearity_params)

    # Combine classified drivers into a single driver group
    drivers_info = accounts_drivers_info.classified_drivers.combine()

    arr_collinearity_dict: dict[dts.AccountType, ArrayF] = {}

    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        all_drivers_lags: dict[dts.DriverName, int] = {}
        for (
            classification
        ) in accounts_drivers_info.classified_drivers.classification_groups.keys():
            for driver in accounts_drivers_info.classified_drivers.maps[
                accounts_drivers_info.classified_drivers.classification_groups[
                    classification
                ]
            ].keys():
                all_drivers_lags[driver] = best_lags[account][classification][driver]
        max_lag = max(all_drivers_lags.values())
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
    preselected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}
    for account in accounts_drivers_info.accounts.account_map.keys():
        preselected_drivers[account] = {
            classification: []
            for classification in accounts_drivers_info.classified_drivers.get_ordered_classifications()
        }

    final_selected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}

    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        final_selected_drivers[account] = _select_final_drivers(
            drivers_info=drivers_info,
            classified_driver_rankings=classified_driver_rankings[account],
            da_params=da_params,
            preselected_drivers=preselected_drivers[account],
            arr_collinearity=arr_collinearity_dict[account],
        )

    return final_selected_drivers

from collections import deque

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from .collinearity_util import use_driver

logger = get_logger(__name__)


def select_final_drivers(
    drivers_map: dict[dts.DriverName, int],
    classified_driver_rankings: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ],
    da_params: params.DriverAnalysisParams,
    preselected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ],
    arr_collinearity_dict: dict[dts.AccountType, ArrayF],
) -> tuple[
    dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]],
    dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]],
]:
    """
    Selects the final set of drivers for each account, removing collinear drivers
    from the ranked candidates and ensuring preselected drivers are included.

    Parameters
    ----------
    drivers_map : dict of {dts.DriverName : int}
        Mapping of driver names to their corresponding indices or identifiers.
    classified_driver_rankings : dict of {dts.AccountType : dict of {dts.DriverClassification : dict of {dts.DriverName : int}}}
        Rankings of drivers within each account and classification, where a lower rank indicates higher priority.
    da_params : params.DriverAnalysisParams
        Driver analysis parameters including collinearity settings.
    preselected_drivers : dict of {dts.AccountType : dict of {dts.DriverClassification : list of dts.DriverName}}
        Drivers that are preselected and must be included in the final set for each account and classification.
    arr_collinearity_dict : dict of {dts.AccountType : ArrayF}
        Collinearity matrices for each account, containing collinearity scores between driver pairs.

    Returns
    -------
    tuple[
        dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]],
        dict[dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]],
    ]
        Tuple containing two dictionaries:
        - The first dictionary maps each account and driver classification to the final selected list of driver names.
        - The second dictionary maps each account and driver classification to the list of skipped driver names.
    """

    final_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}
    final_skipped_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}

    for account in classified_driver_rankings.keys():
        final_drivers[account], final_skipped_drivers[account] = (
            select_final_drivers_account(
                drivers_map=drivers_map,
                classified_driver_rankings=classified_driver_rankings[account],
                da_params=da_params,
                preselected_drivers=preselected_drivers[account],
                arr_collinearity=arr_collinearity_dict[account],
            )
        )

    return final_drivers, final_skipped_drivers


def select_final_drivers_account(
    drivers_map: dict[dts.DriverName, int],
    classified_driver_rankings: dict[
        dts.DriverClassification, dict[dts.DriverName, int]
    ],
    da_params: params.DriverAnalysisParams,
    preselected_drivers: dict[dts.DriverClassification, list[dts.DriverName]],
    arr_collinearity: ArrayF,
) -> tuple[
    dict[dts.DriverClassification, list[dts.DriverName]],
    dict[dts.DriverClassification, list[dts.DriverName]],
]:
    """
    Selects the final set of drivers for each classification, removing collinear drivers
    from the ranked candidates and ensuring preselected drivers are included.

    Parameters
    ----------
    drivers_map : dict of {dts.DriverName : int}
        Mapping of driver names to their corresponding indices or identifiers.
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
    tuple[
        dict[dts.DriverClassification, list[dts.DriverName]],
        dict[dts.DriverClassification, list[dts.DriverName]],
    ]
        Tuple containing two dictionaries:
        - The first dictionary maps each driver classification to the final selected list of driver names.
        - The second dictionary maps each driver classification to the list of skipped driver names.

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

    # TODO: Need to add preselected driver support in the config/params
    # Add a preselected drivers key in the config file, and
    # a corresponding key in the params, will need to be on a per
    # driver classification level, and will need to verify correct
    # driver classifications

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
    classified_ordered_drivers: dict[
        dts.DriverClassification, list[dts.DriverName]
    ] = {}

    for classification in classified_driver_rankings.keys():
        classified_orderings[classification] = deque()
        classified_ordered_drivers[classification] = []
        # Add drivers to the queue in order of their rank (lowest rank first)
        for k in range(len(classified_driver_rankings[classification].keys())):
            classified_orderings[classification].append(
                flipped_classified_driver_rankings[classification][k + 1]
            )
            classified_ordered_drivers[classification].append(
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

    classified_skipped_drivers_set: dict[
        dts.DriverClassification, set[dts.DriverName]
    ] = {classification: set() for classification in classified_driver_rankings.keys()}
    classified_skipped_drivers: dict[dts.DriverClassification, list[dts.DriverName]] = {
        classification: [] for classification in classified_driver_rankings.keys()
    }

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
                    driver_ind=drivers_map[driver],
                    used_driver_inds=[
                        drivers_map[_driver] for _driver in _selected_drivers
                    ],
                    threashold_base=da_params.collinearity_params.threashold_base_removal,
                ):
                    classified_selected_drivers[classification].append(driver)
                    _selected_drivers.append(driver)
                else:
                    classified_skipped_drivers_set[classification].add(driver)
            else:
                # If no drivers selected yet, add the first driver unconditionally
                classified_selected_drivers[classification].append(driver)
                _selected_drivers.append(driver)
            c += 1

        if (
            len(classified_selected_drivers[classification])
            < da_params.n_final_drivers_per_classification
        ):
            logger.warning(
                f'Only {len(classified_selected_drivers[classification])} drivers were selected for '
                f'Classification {classification}, which is less than the desired '
                f'{da_params.n_final_drivers_per_classification}. This may indicate that many drivers are collinear.'
            )
            classified_selected_drivers[classification] = (
                _ensure_correct_num_drivers_selected(
                    drivers_map=drivers_map,
                    selected_drivers=_selected_drivers,
                    selected_drivers_class=classified_selected_drivers[classification],
                    skipped_drivers_class=classified_skipped_drivers_set[
                        classification
                    ],
                    class_drivers=classified_ordered_drivers[classification],
                    da_params=da_params,
                    arr_collinearity=arr_collinearity,
                )
            )

        classified_skipped_drivers[classification] = list(
            classified_skipped_drivers_set[classification]
        )

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
    return classified_selected_drivers, classified_skipped_drivers


def _ensure_correct_num_drivers_selected(
    drivers_map: dict[dts.DriverName, int],
    selected_drivers: list[dts.DriverName],
    selected_drivers_class: list[dts.DriverName],
    skipped_drivers_class: set[dts.DriverName],
    class_drivers: list[dts.DriverName],
    da_params: params.DriverAnalysisParams,
    arr_collinearity: ArrayF,
):
    threashold_base = da_params.collinearity_params.threashold_base_removal
    max_iter = 10
    multiplier = 1.1
    c = 0
    while (
        len(selected_drivers_class) < da_params.n_final_drivers_per_classification
        and c < max_iter
    ):
        for driver in class_drivers:
            if driver not in selected_drivers_class:
                if use_driver(
                    arr_collinearity=arr_collinearity,
                    driver_ind=drivers_map[driver],
                    used_driver_inds=[
                        drivers_map[_driver] for _driver in selected_drivers
                    ],
                    threashold_base=threashold_base * multiplier ** (c + 1),
                ):
                    selected_drivers_class.append(driver)
                    selected_drivers.append(driver)
                    skipped_drivers_class.discard(driver)
        c += 1

    return selected_drivers_class

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

    preselected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}
    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        preselected_drivers[account] = {
            classification: []
            for classification in accounts_drivers_info.classified_drivers.get_ordered_classifications()
        }

    final_pruned_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}

    for account in accounts_drivers_info.accounts.get_ordered_accounts():
        final_pruned_drivers[account] = _prune_drivers(
            drivers_info=drivers_info,
            arr_collinearity=arr_collinearity_dict[account],
            classified_driver_rankings=classified_driver_rankings[account],
            preselected_drivers=preselected_drivers[account],
        )

    return final_pruned_drivers

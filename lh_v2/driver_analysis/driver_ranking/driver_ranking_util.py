from typing import Sequence

import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes import DriverName
from lh_v2.shared import ArrayF, ArrayI


def set_lags_to_zero(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
) -> dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]]:
    """
    Initialize lag values to zero for all drivers across all accounts and classifications.

    This function creates a nested dictionary structure with lag values set to zero
    for every driver in the provided account-driver groups. This is typically used
    as a default initialization before lag optimization.

    Parameters
    ----------
    accounts_drivers_info : dts.AccountGroupClassifiedDriverGroups
        Container with account groups and their associated classified driver groups.

    Returns
    -------
    dict[AccountType, dict[DriverClassification, dict[DriverName, int]]]
        A nested dictionary mapping account types to driver classifications to
        driver names, with all lag values initialized to 0.

    Examples
    --------
    >>> best_lags = set_lags_to_zero(accounts_drivers_info)
    >>> best_lags[AccountType('REVENUE')][DriverClassification.PRIMARY]['driver1']
    0
    """
    # Initialize nested dictionary to store lag values for all accounts and drivers
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ] = {}

    # Iterate through each account type in the accounts structure
    for account_type in accounts_drivers_info.accounts.get_ordered_accounts():
        # Initialize dictionary for this account's driver classifications
        best_lags[account_type] = {}

        # Iterate through each driver classification
        for (
            class_
        ) in accounts_drivers_info.classified_drivers.get_ordered_classifications():
            # Initialize dictionary for drivers in this classification
            best_lags[account_type][class_] = {}

            # Get the index for this classification to access its drivers
            class_idx = accounts_drivers_info.classified_drivers.classification_groups[
                class_
            ]

            # Set lag to 0 for each driver in this classification
            for driver in accounts_drivers_info.classified_drivers.maps[
                class_idx
            ].keys():
                best_lags[account_type][class_][driver] = 0

    return best_lags


def order_drivers_allow_ties(values: ArrayF) -> ArrayI:
    """
    Convert a value array from a driver ranking method to a ranking array.

    Assigns rankings where higher values receive better (lower) ranks. Ties are
    handled by assigning the same rank to drivers with approximately equal values.

    Parameters
    ----------
    values : ArrayF
        A 1D numpy array of values resultant from a driver ranking method.
        For these values, larger values imply better performance.

    Returns
    -------
    ArrayI
        A 1D numpy array where each element represents the ranking position
        of the driver at the corresponding index. Rankings start at 1.

    Notes
    -----
    - Rankings are assigned in descending order of values (highest value = rank 1)
    - Ties are determined using numpy.isclose() with default tolerances
    - Tied drivers receive the same rank, and subsequent ranks continue sequentially

    Examples
    --------
    >>> values = np.array([10.0, 20.0, 20.0, 15.0])
    >>> order_drivers_allow_ties(values)
    array([4, 1, 1, 3], dtype=int32)
    """
    # Sort driver indices by their values in descending order
    # This places the best-performing driver (highest value) first
    inds_sorted = np.argsort(values)[::-1]

    # Initialize output array to store final rankings for each driver
    # Position i contains the rank of driver i
    rankings: ArrayI = np.zeros(values.shape[0], dtype=np.int32)

    # Assign rank 1 to the best-performing driver (first in sorted order)
    rankings[int(inds_sorted[0])] = 1

    # Counter for iterating through remaining drivers in sorted order
    c = 1

    # Process each remaining driver to assign appropriate ranks
    while c < values.shape[0]:
        # Check if current driver's value is approximately equal to previous driver
        # If so, they should receive the same rank (tied performance)
        if np.isclose(values[inds_sorted[c]], values[inds_sorted[c - 1]]):
            # Tie detected: assign same rank as previous driver
            rankings[int(inds_sorted[c])] = rankings[int(inds_sorted[c - 1])]
        else:
            # No tie: assign next sequential rank based on sorted position
            # Note: c+1 ensures proper ranking even after ties
            rankings[int(inds_sorted[c])] = c + 1

        # Move to next driver in sorted order
        c += 1

    return rankings


def order_drivers_no_ties(arr_avg_rankings: ArrayF) -> ArrayI:
    """
    Convert a ranking value array to a ranking array without allowing ties.

    Assigns rankings where lower values receive better (lower) ranks. When drivers
    have approximately equal ranking values, ties are broken deterministically by
    sorting their indices numerically to ensure consistent ordering.

    Parameters
    ----------
    arr_avg_rankings : ArrayF
        A 1D numpy array of ranking values for each driver.
        For these values, smaller values imply better performance.

    Returns
    -------
    ArrayI
        A 1D numpy array where each element represents the ranking position
        of the driver at the corresponding index. Rankings start at 1.
        No two drivers will have the same rank.

    Notes
    -----
    - Rankings are assigned in ascending order of values (lowest value = rank 1)
    - Ties are determined using numpy.isclose() with atol=1e-5
    - When ties occur, drivers are ordered by their index position
    - Tied drivers receive sequential ranks in index order

    Examples
    --------
    >>> values = np.array([2.0, 1.0, 1.0, 3.0])
    >>> order_drivers_no_ties(values)
    array([3, 1, 2, 4], dtype=int32)
    """
    # Sort driver indices by their ranking values in ascending order
    # This places the best-performing driver (lowest ranking value) first
    inds_sorted = np.argsort(arr_avg_rankings)

    # Track the index of the first driver in a potential tie group
    last_ind = 0

    # Dictionary to store tie groups: key is the first index in the tie group,
    # value is a list of subsequent indices that are tied with it
    same_vals: dict[int, list[int]] = {}

    # Scan through sorted indices to identify all tie groups
    for k in range(1, arr_avg_rankings.shape[0]):
        # Check if current driver's ranking value is approximately equal to
        # the value of the first driver in the current tie group
        if np.isclose(
            arr_avg_rankings[inds_sorted[k]],
            arr_avg_rankings[inds_sorted[last_ind]],
            atol=1e-5,  # Consider values within 1e-5 as equal
        ):
            # Add index to the current tie group
            same_vals[last_ind] = same_vals.get(last_ind, []) + [k]
        else:
            # Values differ, so start tracking a new potential tie group
            last_ind = k

    # Initialize output array to store final rankings for each driver
    # Position i contains the rank of driver i
    rankings_arr: ArrayI = np.zeros(arr_avg_rankings.shape[0], dtype=np.int32)

    # Counter for iterating through positions in the sorted array
    c = 0

    # Assign rankings to all drivers, breaking ties deterministically
    while c < arr_avg_rankings.shape[0]:
        if c in same_vals.keys():
            # Tie detected: get all original indices involved in this tie
            # Sort by original index to ensure deterministic tie-breaking
            drivers_idxs: list[int] = sorted(
                [inds_sorted[k] for k in [c] + same_vals[c]]
            )
            # Assign sequential ranks to tied drivers in index order
            # This ensures consistent ordering independent of initial driver order
            for i, driver_idx in enumerate(drivers_idxs):
                rankings_arr[driver_idx] = c + i + 1
            # Skip all tied positions and continue after the tie group
            c += len(same_vals[c]) + 1
        else:
            # No tie: assign rank directly based on sorted position
            rankings_arr[inds_sorted[c]] = c + 1
            c += 1

    return rankings_arr


def rank_array_to_ranking_dict(
    rank_array: ArrayI, ordered_drivers: Sequence[DriverName]
) -> dict[DriverName, int]:
    """
    Convert a rank array to a dictionary mapping driver names to rankings.

    This utility function transforms an array of rankings (where the index
    corresponds to a driver's position) into a more accessible dictionary
    format that directly maps driver names to their ranks.

    Parameters
    ----------
    rank_array : ArrayI
        A 1D numpy array where each element represents the rank of the driver
        at the corresponding index. Rankings typically start at 1.
    ordered_drivers : Sequence[DriverName]
        A sequence of driver names ordered by their indices, corresponding to
        the positions in rank_array.

    Returns
    -------
    dict[DriverName, int]
        A dictionary mapping driver names to their ranking positions.

    Examples
    --------
    >>> rank_array = np.array([2, 1, 3], dtype=np.int32)
    >>> drivers = ["Hamilton", "Verstappen", "Leclerc"]
    >>> rank_array_to_ranking_dict(rank_array, drivers)
    {'Hamilton': 2, 'Verstappen': 1, 'Leclerc': 3}
    """
    # Initialize output dictionary to store driver name to rank mappings
    ranking_dict: dict[DriverName, int] = {}

    # Iterate through drivers and their corresponding ranks
    for idx, driver in enumerate(ordered_drivers):
        # Map each driver name to its rank from the array, converting to int
        ranking_dict[driver] = int(rank_array[idx])

    return ranking_dict


def order_drivers_allow_ties_old(
    values: ArrayF, ordered_drivers: Sequence[DriverName]
) -> dict[DriverName, int]:
    """
    Convert a value array from a driver ranking method to a ranking dictionary.

    This is a legacy version of the ranking function that directly returns a
    dictionary mapping driver names to ranks. Ties are handled by assigning
    the same rank to drivers with approximately equal values.

    Parameters
    ----------
    values : ArrayF
        A 1D numpy array of values resultant from a driver ranking method.
        For these values, larger values imply better performance.
    ordered_drivers : Sequence[DriverName]
        A sequence of driver names ordered by their indices in the values array.

    Returns
    -------
    dict[DriverName, int]
        A dictionary mapping driver names to their ranking positions, where
        rank 1 is the best. Tied drivers receive the same rank.

    Notes
    -----
    This is a legacy function. Consider using `order_drivers_allow_ties` combined
    with `rank_array_to_ranking_dict` for better separation of concerns.

    See Also
    --------
    order_drivers_allow_ties : Modern array-based version of this function.
    rank_array_to_ranking_dict : Converts rank arrays to dictionaries.
    """
    # Sort indices by values in descending order (highest values first)
    inds_sorted = np.argsort(values)[::-1]

    # Initialize ranking dictionary with the highest-valued driver at rank 1
    rank_dict = {ordered_drivers[int(inds_sorted[0])]: 1}

    # Counter to track position in the sorted array
    c = 1

    # Process remaining drivers
    while c < values.shape[0]:
        # Check if current value is approximately equal to the previous value
        if np.isclose(values[inds_sorted[c]], values[inds_sorted[c - 1]]):
            # If values are equal, assign the same rank as the previous driver
            # (tied ranking)
            rank_dict[ordered_drivers[int(inds_sorted[c])]] = rank_dict.get(
                ordered_drivers[int(inds_sorted[c - 1])], c + 1
            )
        else:
            # If values differ, assign rank based on position (c + 1)
            rank_dict[ordered_drivers[int(inds_sorted[c])]] = c + 1

        # Move to next driver in sorted list
        c += 1

    return rank_dict


def order_drivers_no_ties_old(
    arr_avg_rankings: ArrayF, ordered_drivers: Sequence[DriverName]
) -> dict[DriverName, int]:
    """
    Order drivers based on average ranking without allowing ties.

    This is a legacy version that directly returns a dictionary. When drivers have
    approximately equal ranking values, ties are broken deterministically by sorting
    driver names alphabetically to ensure consistent ordering.

    Parameters
    ----------
    arr_avg_rankings : ArrayF
        Array of the average ranking for each driver, where each index corresponds
        to a driver's position in ordered_drivers. Lower values indicate better
        performance.
    ordered_drivers : Sequence[DriverName]
        Sequence of driver names ordered by their indices in arr_avg_rankings.

    Returns
    -------
    dict[DriverName, int]
        A dictionary mapping driver names to their final ranking positions, starting
        at 1. No two drivers will have the same rank; ties are broken alphabetically.

    Notes
    -----
    This function produces deterministic ordering for driver average ranking ties
    independent of driver order by sorting driver names alphabetically. This ensures
    driver selection is a deterministic system independent of the ordering drivers
    are entered in.

    This is a legacy function. Consider using `order_drivers_no_ties` combined
    with `rank_array_to_ranking_dict` for better separation of concerns.

    See Also
    --------
    order_drivers_no_ties : Modern array-based version of this function.
    rank_array_to_ranking_dict : Converts rank arrays to dictionaries.
    """
    # Sort the indices based on the average rankings (ascending order)
    inds_sorted = np.argsort(arr_avg_rankings)
    last_ind = 0
    same_vals: dict[int, list[int]] = {}  # Dictionary to track ties in rankings

    # Find all ties (drivers with very similar average rankings)
    for k in range(1, arr_avg_rankings.shape[0]):
        if np.isclose(
            arr_avg_rankings[inds_sorted[k]],
            arr_avg_rankings[inds_sorted[last_ind]],
            atol=1e-5,  # Consider values within 1e-5 as equal
        ):
            # Add index to the group of tied drivers
            same_vals[last_ind] = same_vals.get(last_ind, []) + [k]
        else:
            # Start a new potential tie group
            last_ind = k

    rankings: dict[DriverName, int] = {}
    c = 0

    # Assign rankings to drivers, handling ties appropriately
    while c < arr_avg_rankings.shape[0]:
        if c in same_vals.keys():
            # Handle ties: get driver names for all tied indices
            drivers = sorted(
                [ordered_drivers[inds_sorted[k]] for k in [c] + same_vals[c]]
            )
            # Assign sequential ranks to tied drivers (alphabetical order)
            for i, driver in enumerate(drivers):
                rankings[driver] = c + i + 1
            # Skip all tied indices
            c += len(same_vals[c]) + 1
        else:
            # No tie, assign rank directly
            rankings[ordered_drivers[inds_sorted[c]]] = c + 1
            c += 1

    return rankings  # Return final driver rankings

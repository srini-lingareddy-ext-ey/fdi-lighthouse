import datetime

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.util import get_logger

from ..driver_ranking import rank

logger = get_logger(__name__)


def _lag_asserts(
    info: dts.AccountGroupClassifiedDriverGroups, lag_params: params.LagParams
):
    """
    Validate lag parameters against account data constraints.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        The account driver group containing the account data with time series
        information.
    lag_params : params.LagParams
        Lag parameters containing n_max_lag and other lag-related settings.

    Raises
    ------
    AssertionError
        If n_max_lag is negative.
    AssertionError
        If the time series length is not sufficient to support the specified
        max_lag value (must be at least max_lag + 2).

    Notes
    -----
    This function performs validation checks to ensure that:
    - The maximum lag parameter is non-negative
    - The time series has enough data points to accommodate the specified lag
    """
    assert lag_params.n_max_lag >= 0, (
        'Cannot have a max lag less than zero, '
        'disable lagging if you wish to not find best lags.'
    )
    assert info.accounts.arr.shape[1] > lag_params.n_max_lag + 1, (
        f'Time series length ({info.accounts.arr.shape[1]} not large '
        f'enough to support max_lag ({lag_params.n_max_lag})).'
    )
    return


def select_best_lags(
    info: dts.AccountGroupClassifiedDriverGroups,
    da_params: params.DriverAnalysisParams,
) -> dict[dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]]:
    """
    Select optimal lag values for each driver based on ranking performance.

    Parameters
    ----------
    info : dts.AccountDriverGroup
        The account driver group containing account data and driver time series.
    lh_params : params.LighthouseParams
        Lighthouse parameters including lag, ranking, and testing configurations.

    Returns
    -------
    dict[dts.DriverName, int]
        A dictionary mapping each driver name to its optimal lag value.
        Returns lag of 0 for all drivers if lagging is disabled or max_lag is 0.

    Notes
    -----
    This function determines the best lag for each driver by:
    1. Creating lagged versions of each driver (from 0 to n_max_lag)
    2. Ranking all lagged versions using the ranking algorithm
    3. Selecting the highest lag among the top n_top_considered performers

    The function returns a lag of 0 for all drivers when lagging is disabled
    (b_lag=False) or when n_max_lag is set to 0.
    """
    logger.info('Selecting best lags for drivers for each account.')
    # Validate that lag parameters are compatible with the time series data
    _lag_asserts(info=info, lag_params=da_params.lag_params)

    # Initialize dictionary to store best lag for each account/classification/driver
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ] = {}

    # If lagging is disabled or max_lag is 0, return zero lag for all drivers
    if not da_params.lag_params.b_lag or da_params.lag_params.n_max_lag == 0:
        logger.info(
            'Lagging disabled or max_lag set to 0, setting all driver lags to 0.'
        )
        # Iterate through all account types
        for account_type in info.accounts.account_map.keys():
            best_lags[account_type] = {}
            # Iterate through all driver classifications
            for (
                driver_classification
            ) in info.classified_drivers.classification_groups.keys():
                best_lags[account_type][driver_classification] = {}
                # Get the classification index for this group
                class_idx = info.classified_drivers.classification_groups[
                    driver_classification
                ]
                # Set lag to 0 for each driver in this classification
                for driver in info.classified_drivers.maps[class_idx].keys():
                    best_lags[account_type][driver_classification][driver] = 0
        return best_lags

    # Create array to hold all lagged versions of drivers
    # Rows: (n_max_lag + 1) versions per driver
    # Cols: time series length reduced by max_lag
    lagged_arr = np.zeros(
        (
            (da_params.lag_params.n_max_lag + 1) * info.classified_drivers.arr.shape[0],
            info.accounts.arr.shape[1] - da_params.lag_params.n_max_lag,
        ),
        dtype=info.np_dtype,
    )

    # Collect all driver names across all classification groups
    all_driver_names: list[dts.DriverName] = []
    for drivers in info.classified_drivers.maps:
        all_driver_names += list(drivers.keys())

    # Create new classification groups treating each original driver as a separate group
    new_classification_groups: dict[dts.DriverClassification, int] = {
        dts.DriverClassification(driver): idx
        for idx, driver in enumerate(all_driver_names)
    }

    # Initialize lists for new maps and dates for each lagged driver version
    new_maps: list[dict[dts.DriverName, int]] = []
    new_dates: list[dict[datetime.date, int]] = []
    c = 0  # Counter for indexing into lagged_arr

    # For each driver, create all lagged versions (0 to n_max_lag)
    for driver in all_driver_names:
        # Create a map for this driver's lagged versions (0, 1, 2, ..., n_max_lag)
        new_maps.append(
            {
                dts.DriverName(str(idx)): idx + c
                for idx in range(da_params.lag_params.n_max_lag + 1)
            }
        )
        # Get the original index of this driver in the classified_drivers array
        old_driver_idx = info.classified_drivers.get_all_ordered_drivers().index(driver)

        # Create each lagged version (lag k means shift back by k time steps)
        for k in range(da_params.lag_params.n_max_lag + 1):
            # Extract the lagged time series for this driver
            # Start at (n_max_lag - k) and end at (shape[1] - k) to align properly
            lagged_arr[c + k, :] = info.classified_drivers.arr[
                old_driver_idx,
                da_params.lag_params.n_max_lag - k : info.accounts.arr.shape[1] - k,
            ]
            # Copy date information for this lagged version
            new_dates.append(info.classified_drivers.dates[old_driver_idx])

        # Move counter forward by number of lagged versions per driver
        c += da_params.lag_params.n_max_lag + 1

    ranking_params_c = da_params.ranking_params.model_copy()
    ranking_params_c.methods = da_params.lag_params.ranking_methods

    # Rank all lagged driver versions using the ranking algorithm
    lagged_rankings = rank(
        accounts_drivers_info=dts.AccountGroupClassifiedDriverGroups(
            # Apply lag to accounts to match shortened time series
            accounts=info.accounts.apply_lag(max_lag=da_params.lag_params.n_max_lag),
            # Use the newly created lagged driver data
            classified_drivers=dts.ClassifiedDriverGroups(
                arr=lagged_arr,
                classification_groups=new_classification_groups,
                maps=new_maps,
                dates=new_dates,
                np_dtype=info.np_dtype,
            ),
            np_dtype=info.np_dtype,
        ),
        ranking_params=ranking_params_c,
        logger=logger,
    )

    # Initialize dictionary for storing best lags
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ] = {}

    # For each account type, determine the best lag for each driver
    for account in lagged_rankings.keys():
        best_lags[account] = {}
        # Iterate through original driver classifications
        for (
            driver_classification
        ) in info.classified_drivers.classification_groups.keys():
            best_lags[account][driver_classification] = {}

            # Collect all drivers in this classification
            drivers = []
            for driver in info.classified_drivers.maps[
                info.classified_drivers.classification_groups[driver_classification]
            ].keys():
                drivers.append(driver)

            # For each driver, find the best lag value
            for driver in drivers:
                # Invert the ranking dictionary to map rank -> lag value
                driver_lags_flipped = {
                    val: key for key, val in lagged_rankings[account][driver].items()
                }
                # Select the highest lag among the top n_top_considered ranked lags
                best_lags[account][driver_classification][driver] = sorted(
                    [
                        int(driver_lags_flipped[rank_idx])
                        for rank_idx in range(
                            1, da_params.lag_params.n_top_considered + 1
                        )
                    ]
                )[-1]

    return best_lags

import datetime
from dataclasses import dataclass

import numpy as np
from dateutil.relativedelta import relativedelta

from lh_v2.shared import BASE_NP_DTYPE

from .account_types import (
    AccountGroupInfo,
    AccountInfo,
    AccountType,
)
from .driver_types import (
    ClassifiedDriverGroups,
    DriverClassification,
    DriverGroup,
    DriverName,
)


@dataclass
class AccountDriverGroup:
    """
    Represents a grouping of an account and its associated driver group.

    Attributes
    ----------
    account : AccountInfo
        The account information associated with this group.
    drivers : DriverGroup
        The group of drivers associated with the account.
    np_dtype : type, default=BASE_NP_DTYPE
        The NumPy data type used for numerical operations.

    Methods
    -------
    apply_lags(best_lags, max_lag)
        Apply lag transformations to both account and drivers.
    apply_daterange(start_date, end_date, best_lags, b_training)
        Apply date range filter with lag adjustments to account and drivers.
    """

    account: AccountInfo
    drivers: DriverGroup
    np_dtype: type = BASE_NP_DTYPE

    def apply_lags(
        self, best_lags: dict[DriverName, int], max_lag: int
    ) -> AccountDriverGroup:
        """
        Apply lag transformations to both account and drivers.

        Parameters
        ----------
        best_lags : dict[DriverName, int]
            Mapping of driver names to their optimal lag values.
        max_lag : int
            Maximum lag value to apply.

        Returns
        -------
        AccountDriverGroup
            A new instance with lagged account and driver data.
        """
        # Create new instance with both account and drivers lagged for temporal alignment
        return AccountDriverGroup(
            account=self.account.apply_lag(max_lag=max_lag),  # Lag account by max_lag
            drivers=self.drivers.apply_lags(
                best_lags=best_lags, max_lag=max_lag
            ),  # Lag each driver by its specific lag
            np_dtype=self.np_dtype,
        )

    def _apply_daterange_asserts(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        best_lags: dict[DriverName, int],
        b_training: bool,
    ) -> None:
        """
        Validate date ranges for training or prediction with driver lags.

        This method performs assertions to ensure that the specified date range is valid
        for the account and that driver data is available for the required lag periods.

        Parameters
        ----------
        start_date : datetime.date
            The starting date of the date range to validate.
        end_date : datetime.date
            The ending date of the date range to validate.
        best_lags : dict[DriverName, int]
            A dictionary mapping driver names to their optimal lag values (in months).
        b_training : bool
            Flag indicating whether validation is for training (True) or prediction (False).
            When True, validates that the date range is long enough for the maximum lag.
            When False, validates that driver data exists for all lagged periods.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If start_date is not in the account's date range.
        AssertionError
            If end_date is not in the account's date range.
        AssertionError
            If b_training is True and the date range is shorter than the maximum lag.
        AssertionError
            If b_training is False and any driver's lagged start date is not in the
            driver's date range.
        AssertionError
            If b_training is False and any driver's lagged end date is not in the
            driver's date range.

        Notes
        -----
        This is a private method used internally by apply_daterange to validate inputs
        before applying date range transformations.
        """
        # Verify that the start date exists in the account's available dates
        assert start_date in self.account.dates.keys(), (
            'Start date is not in account date range.'
        )
        # Verify that the end date exists in the account's available dates
        assert end_date in self.account.dates.keys(), (
            'End date is not in account date range.'
        )

        # Training-specific validation: ensure sufficient data for maximum lag
        if b_training:
            max_lag = max(best_lags.values())
            # Calculate the number of periods in the date range
            # Must be at least as long as the maximum lag to allow proper training
            assert (
                self.account.dates[end_date] - self.account.dates[start_date] + 1
                >= max_lag
            ), (
                'The given date range is too short to accommodate the maximum lag for training.'
            )

        # Prediction-specific validation: ensure driver data exists for all lagged periods
        if not b_training:
            for driver, lag in best_lags.items():
                # Calculate the lagged start date by subtracting the driver's lag
                driver_start_date = start_date + relativedelta(months=-lag)
                # Calculate the lagged end date by subtracting the driver's lag
                driver_end_date = end_date + relativedelta(months=-lag)
                # Verify the lagged start date exists in this driver's date range
                assert (
                    driver_start_date in self.drivers.dates[self.drivers.map[driver]]
                ), f'Start date for driver {driver} is not in driver date range.'
                # Verify the lagged end date exists in this driver's date range
                assert (
                    driver_end_date in self.drivers.dates[self.drivers.map[driver]]
                ), f'End date for driver {driver} is not in driver date range.'

        return

    def apply_daterange(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        best_lags: dict[DriverName, int],
        b_training: bool,
    ) -> AccountDriverGroup:
        """
        Apply a date range filter to the account and drivers with lag adjustments.

        This method filters both the account and driver data to specified date ranges,
        accounting for time lags between drivers and the target variable. During training,
        the start date is adjusted forward by the maximum lag to ensure proper alignment.

        Parameters
        ----------
        start_date : datetime.date
            The starting date for the account data range.
        end_date : datetime.date
            The ending date for the account data range.
        best_lags : dict[DriverName, int]
            Dictionary mapping each driver name to its optimal lag in months.
            Positive lag values indicate the driver leads the account data.
        b_training : bool
            Flag indicating whether this is for training data. If True, the start_date
            is shifted forward by the maximum lag value.

        Returns
        -------
        AccountDriverGroup
            A new AccountDriverGroup instance with date-filtered account and driver data.

        Notes
        -----
        - Currently only supports monthly time series values.
        - Driver date ranges are adjusted backwards by their respective lags to align
          with the account data timeline.
        - During training (b_training=True), the account start_date is moved forward
          by max_lag months to ensure all lagged drivers have sufficient history.

        Examples
        --------
        >>> best_lags = {'driver1': 1, 'driver2': 3}
        >>> filtered = account_group.apply_daterange(
        ...     start_date=date(2023, 1, 1),
        ...     end_date=date(2023, 12, 31),
        ...     best_lags=best_lags,
        ...     b_training=True
        ... )
        """
        # Validate that date ranges are valid and driver data exists for lagged periods
        # TODO: this currently only works if time series values are monthly
        self._apply_daterange_asserts(
            start_date=start_date,
            end_date=end_date,
            best_lags=best_lags,
            b_training=b_training,
        )

        # For training, shift account start date forward by maximum lag
        # This ensures we have sufficient driver history for all lag periods
        if b_training:
            max_lag = max(best_lags.values())
            start_date = start_date + relativedelta(months=max_lag)

        # Calculate driver-specific start dates by shifting back by each driver's lag
        # This aligns driver data to match the account's target timeline
        # Example: if account starts at 2023-06-01 and driver has lag=3,
        # driver data should start at 2023-03-01 to predict 2023-06-01
        driver_start_dates = {
            driver: start_date + relativedelta(months=-lag)
            for driver, lag in best_lags.items()
        }

        # Calculate driver-specific end dates by shifting back by each driver's lag
        driver_end_dates = {
            driver: end_date + relativedelta(months=-lag)
            for driver, lag in best_lags.items()
        }

        # Create new AccountDriverGroup with filtered date ranges for account and drivers
        return AccountDriverGroup(
            account=self.account.apply_daterange(
                start_date=start_date, end_date=end_date
            ),
            drivers=self.drivers.apply_daterange(
                start_dates=driver_start_dates, end_dates=driver_end_dates
            ),
            np_dtype=self.np_dtype,
        )


@dataclass
class AccountClassifiedDriverGroups:
    """
    A container for account information and classified driver groups.

    This class manages the relationship between an account and its associated
    driver groups, organized by classification. It provides functionality for
    accessing driver groups by classification and applying lag transformations
    to both account and driver data.

    Attributes
    ----------
    account : AccountInfo
        The account information associated with the driver groups.
    classified_drivers : ClassifiedDriverGroups[DriverClassification]
        The driver groups organized by classification.
    np_dtype : type, default=BASE_NP_DTYPE
        The numpy data type to use for numerical operations.

    Methods
    -------
    __getitem__(key)
        Retrieve an AccountDriverGroup for a specific driver classification.
    apply_lags(best_lags, max_lag)
        Apply lag transformations to both account and driver data.

    Examples
    --------
    >>> group = account_classified_drivers[DriverClassification.PRIMARY]
    >>> lagged = account_classified_drivers.apply_lags(best_lags, max_lag=12)
    """

    account: AccountInfo
    classified_drivers: ClassifiedDriverGroups[DriverClassification]
    np_dtype: type = BASE_NP_DTYPE

    def __getitem__(self, key: DriverClassification) -> AccountDriverGroup:
        """
        Retrieve an AccountDriverGroup for a specific driver classification.

        Parameters
        ----------
        key : DriverClassification
            The classification key used to retrieve the corresponding driver group.

        Returns
        -------
        AccountDriverGroup
            An account driver group containing the account and drivers matching
            the specified classification.

        Raises
        ------
        KeyError
            If the classification key does not exist in classified_drivers.
        """
        # Create AccountDriverGroup combining account with specific classification's drivers
        return AccountDriverGroup(
            account=self.account,
            drivers=self.classified_drivers[
                key
            ],  # Extract drivers for this classification
        )

    def apply_lags(
        self, best_lags: dict[DriverClassification, dict[DriverName, int]], max_lag: int
    ) -> AccountClassifiedDriverGroups:
        """
        Apply lag transformations to both account and classified driver data.

        This method creates a new instance with the account and all classified
        drivers shifted according to their specified lag values.

        Parameters
        ----------
        best_lags : dict[DriverClassification, dict[DriverName, int]]
            Nested mapping of driver classifications to their respective driver
            names and optimal lag values. Each lag value represents the number
            of time periods to shift that driver backward.
        max_lag : int
            Maximum lag value to apply to the account and driver data. The
            resulting time series will be shortened by this amount.

        Returns
        -------
        AccountClassifiedDriverGroups
            A new instance with lagged account and classified driver data. The
            time dimension is reduced by max_lag to ensure alignment.

        Notes
        -----
        Both the account and all drivers are lagged to maintain temporal alignment.
        The account is lagged by max_lag, while each driver is lagged according to
        its specific value in best_lags.
        """
        # Create new instance with lagged account and all classified driver groups
        return AccountClassifiedDriverGroups(
            account=self.account.apply_lag(max_lag=max_lag),  # Lag account uniformly
            classified_drivers=self.classified_drivers.apply_lag(
                best_lags=best_lags,
                max_lag=max_lag,  # Lag each driver by its classification-specific lag
            ),
            np_dtype=self.np_dtype,
        )


@dataclass
class AccountGroupDriverGroup:
    """
    A container class that manages account group information and driver group data.

    This class pairs account group information with driver groups and provides
    functionality for indexing by account type and applying lag transformations.

    Attributes
    ----------
    accounts : AccountGroupInfo
        Information about the account group.
    drivers : DriverGroup
        Group of drivers associated with the accounts.
    np_dtype : type, default=BASE_NP_DTYPE
        NumPy data type to use for numerical operations.

    Methods
    -------
    __getitem__(key: AccountType) -> AccountDriverGroup
        Retrieve an AccountDriverGroup for a specific account type.
    apply_lags(best_lags: dict[DriverName, int], max_lag: int) -> AccountGroupDriverGroup
        Apply lag transformations to both accounts and drivers.

    Examples
    --------
    >>> ag_driver_group = AccountGroupDriverGroup(accounts=account_info, drivers=driver_grp)
    >>> specific_account = ag_driver_group[AccountType.SAVINGS]
    >>> lagged_group = ag_driver_group.apply_lags(best_lags={'driver1': 2}, max_lag=5)
    """

    accounts: AccountGroupInfo
    drivers: DriverGroup
    np_dtype: type = BASE_NP_DTYPE

    def __getitem__(self, key: AccountType) -> AccountDriverGroup:
        """
        Retrieve an AccountDriverGroup for a specific account type.

        Parameters
        ----------
        key : AccountType
            The account type to retrieve the driver group for.

        Returns
        -------
        AccountDriverGroup
            An AccountDriverGroup instance containing the account corresponding to the
            specified account type and the associated drivers.

        Raises
        ------
        KeyError
            If the specified account type is not found in the accounts dictionary.
        """
        # Create AccountDriverGroup pairing specific account with all drivers
        return AccountDriverGroup(account=self.accounts[key], drivers=self.drivers)

    def apply_lags(
        self, best_lags: dict[DriverName, int], max_lag: int
    ) -> AccountGroupDriverGroup:
        """
        Apply lag transformations to both account group and driver data.

        This method creates a new instance where all accounts and drivers have
        been shifted according to their specified lag values to maintain temporal
        alignment.

        Parameters
        ----------
        best_lags : dict[DriverName, int]
            Mapping of driver names to their optimal lag values. Each value
            represents the number of time periods to shift that driver backward.
        max_lag : int
            Maximum lag value to apply to the account and driver data. All
            accounts are shifted by this amount, and the resulting time series
            will be shortened by max_lag.

        Returns
        -------
        AccountGroupDriverGroup
            A new instance with lagged account group and driver data. The time
            dimension is reduced by max_lag to ensure all series have valid data.

        Notes
        -----
        The accounts are lagged uniformly by max_lag, while drivers are lagged
        individually according to best_lags values.
        """
        # Create new instance with lagged accounts and drivers
        return AccountGroupDriverGroup(
            accounts=self.accounts.apply_lag(
                max_lag=max_lag
            ),  # Lag all accounts uniformly
            drivers=self.drivers.apply_lags(
                best_lags=best_lags, max_lag=max_lag
            ),  # Lag each driver individually
            np_dtype=self.np_dtype,
        )


@dataclass
class AccountGroupSelectedDrivers:
    """
    A container for account groups and their associated classified driver groups.

    This class pairs account group information with classified driver groups,
    ensuring that the account types match between the two structures. It is
    typically created by the select_drivers method of AccountGroupClassifiedDriverGroups.

    Attributes
    ----------
    accounts : AccountGroupInfo
        Information about account groups, containing a mapping of account types.
    drivers : ClassifiedDriverGroups[AccountType]
        Classified driver groups organized by account type, where each account
        type has its own set of selected drivers.
    np_dtype : type, optional
        NumPy data type to use for numerical operations. Defaults to BASE_NP_DTYPE.

    Methods
    -------
    __getitem__(key: AccountType) -> AccountDriverGroup
        Retrieve an AccountDriverGroup for a specific account type.

    Raises
    ------
    AssertionError
        If the account types in drivers do not match those in accounts.account_map.

    Notes
    -----
    The __post_init__ method validates that account types are consistent between
    the accounts and drivers structures. This ensures that each account type has
    a corresponding set of drivers.

    Examples
    --------
    >>> selected = account_group_classified.select_drivers(driver_selection)
    >>> account_drivers = selected[AccountType.REVENUE]
    """

    accounts: AccountGroupInfo
    drivers: ClassifiedDriverGroups[AccountType]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        """Validate that account types match between accounts and drivers."""
        # Ensure structural consistency between accounts and their associated drivers
        assert set(self.accounts.account_map.keys()) == set(
            self.drivers.classification_groups.keys()
        ), 'Account types in drivers must match those in accounts.account_map.'
        return

    def __getitem__(self, key: AccountType) -> AccountDriverGroup:
        """
        Retrieve an AccountDriverGroup for a specific account type.

        Parameters
        ----------
        key : AccountType
            The account type to retrieve the driver group for.

        Returns
        -------
        AccountDriverGroup
            An AccountDriverGroup instance containing the specified account and
            its associated selected drivers.

        Raises
        ------
        AssertionError
            If the specified account type is not found in the accounts.

        Examples
        --------
        >>> revenue_group = selected_drivers[AccountType('REVENUE')]
        """
        # Validate that the requested account type exists
        assert key in self.accounts.account_map, (
            f'Account type {key} not found in accounts.'
        )
        # Create AccountDriverGroup pairing specific account with its selected drivers
        return AccountDriverGroup(
            account=self.accounts[key],
            drivers=self.drivers[key],  # Drivers already filtered for this account type
            np_dtype=self.np_dtype,
        )


@dataclass
class AccountGroupClassifiedDriverGroups:
    """
    Container for account group information and classified driver groups.

    This class associates account group information with classified driver groups,
    providing indexed access to individual account-classified driver combinations
    and methods for driver selection and date range filtering.

    Attributes
    ----------
    accounts : AccountGroupInfo
        Information about the account groups.
    classified_drivers : ClassifiedDriverGroups[DriverClassification]
        Groups of classified drivers, organized by classification (e.g., economic
        indicators, technical factors).
    np_dtype : type, optional
        NumPy data type for numerical operations, defaults to BASE_NP_DTYPE.

    Methods
    -------
    __getitem__(key)
        Retrieve account-classified driver groups for a specific account type.
    select_drivers(selected_drivers)
        Select specific drivers from classified groups for each account.
    apply_daterange(start_date, end_date)
        Filter both accounts and drivers to a specified date range.

    Examples
    --------
    >>> ag_classified = AccountGroupClassifiedDriverGroups(accounts, drivers)
    >>> account_drivers = ag_classified[AccountType('REVENUE')]
    >>> selected = ag_classified.select_drivers(driver_selection)
    """

    accounts: AccountGroupInfo
    classified_drivers: ClassifiedDriverGroups[DriverClassification]
    np_dtype: type = BASE_NP_DTYPE

    def __getitem__(self, key: AccountType) -> AccountClassifiedDriverGroups:
        """
        Retrieve AccountClassifiedDriverGroups for a specific account type.

        Parameters
        ----------
        key : AccountType
            The account type to retrieve the classified driver groups for.

        Returns
        -------
        AccountClassifiedDriverGroups
            An instance containing the account and its associated classified drivers.
        """
        # Create AccountClassifiedDriverGroups for specific account with all classifications
        return AccountClassifiedDriverGroups(
            account=self.accounts[key],
            classified_drivers=self.classified_drivers,  # All classifications available
        )

    def select_drivers(
        self,
        selected_drivers: dict[
            AccountType, dict[DriverClassification, list[DriverName]]
        ],
    ) -> AccountGroupSelectedDrivers:
        """
        Select specific drivers from classified driver groups and create a new AccountGroupSelectedDrivers instance.

        This method filters drivers based on the provided selection dictionary and creates a new
        consolidated array containing only the selected drivers, maintaining their association with
        account types.

        Parameters
        ----------
        selected_drivers : dict[AccountType, dict[DriverClassification, list[DriverName]]]
            A nested dictionary structure specifying which drivers to select. The outer dictionary
            maps account types to their driver selections. The inner dictionary maps driver
            classifications to lists of driver names to be selected.

        Returns
        -------
        AccountGroupSelectedDrivers
            A new instance containing the selected drivers with updated mappings and arrays.
            The drivers array contains only the selected drivers, with updated index mappings
            for each account type.

        Raises
        ------
        AssertionError
            If the account types in selected_drivers do not match the account types in
            self.accounts.account_map.

        Notes
        -----
        - The method creates a new numpy array to store the selected drivers' data
        - Index mappings are updated to reflect the new positions in the consolidated array
        - The order of drivers in the output follows the order of accounts and the order
          of drivers within each classification
        - Date information is preserved for each selected driver
        """
        # Validate that selected_drivers contains the same account types as the accounts
        assert set(selected_drivers.keys()) == set(self.accounts.account_map.keys()), (
            'Selected drivers account types must match accounts account types.'
        )

        # Calculate the total number of drivers to be selected across all accounts and classifications
        n_drivers_total = 0
        for account in selected_drivers.keys():
            for classification in selected_drivers[account].keys():
                n_drivers_total += len(selected_drivers[account][classification])

        # Create a new array to store the selected drivers' time series data
        # Dimensions: (number of selected drivers, time periods)
        new_drivers_arr = np.zeros(
            (n_drivers_total, self.accounts.arr.shape[1]),
            dtype=self.np_dtype,
        )

        # Initialize containers for the new driver mappings and date indices
        # One map per account type, one date dict per driver
        new_maps: list[dict[DriverName, int]] = []
        new_dates: list[dict[datetime.date, int]] = []
        current_index = 0  # Track position in the new consolidated array

        # Iterate through accounts in their canonical order
        for idx, account in enumerate(self.accounts.get_ordered_accounts()):
            # Initialize a new mapping dictionary for this account type
            new_maps.append({})

            # Iterate through each classification for this account
            for classification in selected_drivers[account].keys():
                # Iterate through each selected driver within this classification
                for driver in selected_drivers[account][classification]:
                    # Look up the driver's index in the original classified drivers array
                    old_index = self.classified_drivers[classification].map[driver]

                    # Copy the driver's time series data to the new consolidated array
                    new_drivers_arr[current_index, :] = self.classified_drivers[
                        classification
                    ].arr[old_index, :]

                    # Preserve the driver's date-to-index mapping
                    new_dates.append(self.classified_drivers.dates[old_index])

                    # Map the driver name to its new index in the consolidated array
                    new_maps[idx][driver] = current_index

                    # Increment the index for the next driver
                    current_index += 1

        # Construct and return a new AccountGroupSelectedDrivers instance
        # with the selected drivers organized by account type
        return AccountGroupSelectedDrivers(
            accounts=self.accounts,
            drivers=ClassifiedDriverGroups[AccountType](
                arr=new_drivers_arr,
                classification_groups=self.accounts.account_map,
                maps=new_maps,
                dates=new_dates,
                np_dtype=self.np_dtype,
            ),
            np_dtype=self.np_dtype,
        )

    def apply_daterange(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> AccountGroupClassifiedDriverGroups:
        """
        Apply a date range filter to both accounts and classified drivers.

        This method creates a new instance containing only data within the
        specified date range for both accounts and all classified drivers.

        Parameters
        ----------
        start_date : datetime.date
            The start date of the range to filter by (inclusive).
        end_date : datetime.date
            The end date of the range to filter by (inclusive).

        Returns
        -------
        AccountGroupClassifiedDriverGroups
            A new instance with the date range applied to both accounts and
            classified drivers. The numpy dtype is preserved.

        Notes
        -----
        Both the accounts and all driver classifications are filtered to the
        same date range. Date indices are reindexed to start from 0 in the
        returned instance.

        Examples
        --------
        >>> filtered = ag_classified.apply_daterange(
        ...     start_date=date(2023, 1, 1),
        ...     end_date=date(2023, 12, 31)
        ... )
        """
        # Create new instance with date-filtered accounts and drivers
        return AccountGroupClassifiedDriverGroups(
            accounts=self.accounts.apply_daterange(
                start_date=start_date,
                end_date=end_date,  # Filter accounts to date range
            ),
            classified_drivers=self.classified_drivers.apply_daterange(
                start_date=start_date,
                end_date=end_date,  # Filter all driver classifications to same date range
            ),
            np_dtype=self.np_dtype,
        )

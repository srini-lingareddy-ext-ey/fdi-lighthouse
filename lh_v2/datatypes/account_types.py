import datetime
from dataclasses import dataclass
from typing import NewType, Optional, Sequence

import numpy as np

from lh_v2.shared import BASE_NP_DTYPE, ArrayF
from lh_v2.util import flip_dict, flip_seq_dicts, month_dif

from .hierarchy_util import HierarchyTree, LocationType, ProductType

AccountType = NewType('AccountType', str)


@dataclass
class AccountInfo:
    """
    Represents account information with associated time series data and metadata.

    This class encapsulates a single account's time series data along with its
    temporal mapping, account classification, and hierarchical categorization.

    Attributes
    ----------
    arr : ArrayF
        A NumPy array containing the time series data for the account.
    dates : dict[datetime.date, int]
        A mapping of dates to their corresponding indices in the `arr`.
    account_type : AccountType
        The type of the account (e.g., 'REVENUE', 'COGS').
    segment_type : HierarchyTree[ProductType]
        The product hierarchy level associated with the account.
    region_type : HierarchyTree[LocationType]
        The location hierarchy level associated with the account.
    np_dtype : type, default=BASE_NP_DTYPE
        The NumPy data type used for the account data.

    Methods
    -------
    get_info() -> tuple[AccountType, ProductType, LocationType]
        Retrieves the account type, segment name, and region name.
    flip_dates() -> dict[int, datetime.date]
        Returns a reversed mapping of indices to dates.
    apply_lag(max_lag: int) -> AccountInfo
        Applies a lag to the account data and returns a new AccountInfo instance.
    apply_daterange(start_date: Optional[datetime.date], end_date: Optional[datetime.date]) -> AccountInfo
        Filters the account data to a specific date range.
    add_forecast_vals(forecast_dates: Sequence[datetime.date], forecast_values: ArrayF) -> AccountInfo
        Adds forecast values to the account data.

    Notes
    -----
    The array data is automatically cast to the specified np_dtype in __post_init__.
    All methods that transform the data return new AccountInfo instances, preserving
    immutability of the original object.
    """

    arr: ArrayF
    dates: dict[datetime.date, int]
    account_type: AccountType
    segment_type: HierarchyTree[ProductType]
    region_type: HierarchyTree[LocationType]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        """Cast array to specified numpy dtype upon initialization."""
        # Ensure array uses consistent dtype for all operations
        self.arr = self.arr.astype(self.np_dtype)
        return

    def get_info(self) -> tuple[AccountType, ProductType, LocationType]:
        """
        Retrieve account classification and hierarchical information.

        Returns
        -------
        tuple[AccountType, ProductType, LocationType]
            A tuple containing:
            - account_type: The type of account (e.g., 'REVENUE', 'COGS')
            - segment_name: The name of the product hierarchy node
            - region_name: The name of the location hierarchy node
        """
        # Extract names from hierarchy tree nodes for easier access
        return self.account_type, self.segment_type.name, self.region_type.name

    def flip_dates(self) -> dict[int, datetime.date]:
        """
        Create a reverse mapping from array indices to dates.

        Returns
        -------
        dict[int, datetime.date]
            A dictionary mapping array indices to their corresponding dates.
            This is the inverse of the `dates` attribute.

        See Also
        --------
        flip_dict : Utility function used to reverse the dictionary.

        Examples
        --------
        >>> account_info.dates
        {datetime.date(2023, 1, 1): 0, datetime.date(2023, 2, 1): 1}
        >>> account_info.flip_dates()
        {0: datetime.date(2023, 1, 1), 1: datetime.date(2023, 2, 1)}
        """
        # Reverse the dates dictionary to enable index-to-date lookups
        return flip_dict(dictionary=self.dates)

    def _lag_dates(self, max_lag: int) -> dict[datetime.date, int]:
        """
        Create a mapping of dates to indices for lagged time series data.

        This method generates a dictionary that maps dates to their corresponding indices
        in a lagged array. It accounts for the loss of observations due to lagging by
        offsetting the date mapping.

        Parameters
        ----------
        max_lag : int
            The maximum number of lag periods to account for. This determines how many
            observations are lost at the beginning of the time series.

        Returns
        -------
        dict[datetime.date, int]
            A dictionary mapping dates to their corresponding indices in the lagged array.
            The dates are offset by max_lag periods, and the mapping excludes the first
            (max_lag + 1) observations.

        Notes
        -----
        The method uses `flip_dates()` to obtain the original date sequence and creates
        a new mapping that accounts for the observations lost due to lagging. For a time
        series with n observations and max_lag periods, the resulting dictionary will
        contain (n - max_lag - 1) entries.
        """
        # Get reverse mapping from indices to dates
        flipped_dates = self.flip_dates()

        # Initialize new date-to-index mapping for lagged data
        new_dates: dict[datetime.date, int] = {}

        # Map dates to new indices, skipping the first (max_lag + 1) observations
        # which are lost due to lagging
        for k in range(self.arr.shape[0] - (max_lag + 1)):
            # Date at position (k + max_lag) in original series maps to index k in lagged series
            new_dates[flipped_dates[k + max_lag]] = k

        return new_dates

    def apply_lag(self, max_lag: int) -> AccountInfo:
        """
        Apply a lag to the account data by removing the first max_lag entries.

        This method creates a new AccountInfo instance with the data shifted by
        removing the first max_lag rows from the array and adjusting the dates
        accordingly.

        Parameters
        ----------
        max_lag : int
            The number of time periods to lag the data. This many rows will be
            removed from the beginning of the data array.

        Returns
        -------
        AccountInfo
            A new AccountInfo instance with lagged data, containing:
            - arr: The original array with the first max_lag rows removed
            - dates: The dates adjusted for the lag period
            - All other attributes (account_type, segment_type, region_type,
              np_dtype) preserved from the original instance
        """
        # Create new instance with sliced array and adjusted dates
        return AccountInfo(
            arr=self.arr[max_lag:],  # Remove first max_lag observations
            dates=self._lag_dates(max_lag=max_lag),  # Adjust date mappings
            account_type=self.account_type,
            segment_type=self.segment_type,
            region_type=self.region_type,
            np_dtype=self.np_dtype,
        )

    def apply_daterange(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> AccountInfo:
        """
        Filter the account data to a specific date range.

        This method creates a new AccountInfo instance containing only the data
        within the specified date range. If start_date or end_date are None,
        the method uses the earliest or latest available dates respectively.

        Parameters
        ----------
        start_date : Optional[datetime.date], default=None
            The starting date for the filtered range (inclusive). If None,
            starts from the beginning of the available data.
        end_date : Optional[datetime.date], default=None
            The ending date for the filtered range (inclusive). If None,
            extends to the end of the available data.

        Returns
        -------
        AccountInfo
            A new AccountInfo instance containing:
            - arr: The filtered array data for the specified date range
            - dates: A re-indexed dictionary mapping dates to their new positions
            - All other attributes (account_type, segment_type, region_type,
              np_dtype) preserved from the original instance

        Raises
        ------
        ValueError
            If start_date is specified but not present in the available dates.
        ValueError
            If end_date is specified but not present in the available dates.

        Notes
        -----
        The date indices in the returned AccountInfo are re-indexed starting from 0
        to reflect the new array positions after filtering.
        """
        # Validate that start_date exists in available dates
        if start_date is not None and start_date not in self.dates.keys():
            raise ValueError('Start date is after available date range.')

        # Validate that end_date exists in available dates
        if end_date is not None and end_date not in self.dates.keys():
            raise ValueError('End date is before available date range.')

        # Determine starting index: use 0 if no start_date specified
        if start_date is None:
            start_idx = 0
        else:
            start_idx = self.dates[start_date]

        # Determine ending index: use array length if no end_date specified
        if end_date is None:
            end_idx = self.arr.shape[0]
        else:
            end_idx = self.dates[end_date] + 1  # +1 to make end_date inclusive

        # Create new AccountInfo with filtered array and re-indexed dates
        return AccountInfo(
            arr=self.arr[start_idx:end_idx],
            dates={
                date: idx - start_idx  # Re-index dates to start from 0
                for date, idx in self.dates.items()
                if start_idx <= idx < end_idx  # Keep only dates within range
            },
            account_type=self.account_type,
            segment_type=self.segment_type,
            region_type=self.region_type,
            np_dtype=self.np_dtype,
        )

    def add_forecast_vals(
        self, forecast_dates: Sequence[datetime.date], forecast_values: ArrayF
    ) -> AccountInfo:
        """
        Add forecast values to the account information, creating a new AccountInfo instance.

        This method combines historical data (up to the first forecast date) with new forecast
        values, creating a continuous time series of account information.

        Parameters
        ----------
        forecast_dates : Sequence[datetime.date]
            A sequence of dates corresponding to the forecast period. Must be in chronological
            order and start after the last historical date.
        forecast_values : ArrayF
            Array of forecast values with length matching forecast_dates. Values will be cast
            to the account's numpy dtype.

        Returns
        -------
        AccountInfo
            A new AccountInfo instance containing both historical data (up to the first forecast
            date) and the provided forecast values.

        Raises
        ------
        AssertionError
            If the length of forecast_dates does not match the number of forecast_values.

        Notes
        -----
        - Historical data before the first forecast date is preserved and copied to the new array.
        - The method creates a new dates dictionary mapping dates to array indices.
        - Both historical and forecast values are cast to the account's numpy dtype.
        """
        # Validate that forecast dates and values have matching lengths
        assert len(forecast_dates) == forecast_values.shape[0], (
            'Length of forecast_dates must match the number of forecast_values'
        )

        # Build new dates dictionary with only historical dates before first forecast date
        new_dates: dict[datetime.date, int] = {
            date: idx for date, idx in self.dates.items() if date < forecast_dates[0]
        }

        # Count historical observations and identify the last historical date
        n_hist = len(new_dates)
        last_hist_date = max(new_dates.keys())

        # Add forecast dates to the dates dictionary, continuing the index sequence
        for idx, date in enumerate(forecast_dates):
            new_dates[date] = n_hist + idx

        # Initialize array to hold both historical and forecast values
        new_arr = np.zeros(
            (len(new_dates),),
            dtype=self.np_dtype,
        )

        # Copy historical data up to the last historical date
        new_arr[:n_hist] = self.apply_daterange(end_date=last_hist_date).arr.astype(
            self.np_dtype
        )

        # Append forecast values to the array
        new_arr[n_hist:] = forecast_values.astype(self.np_dtype)

        # Return new AccountInfo instance with combined historical and forecast data
        return AccountInfo(
            arr=new_arr,
            dates=new_dates,
            account_type=self.account_type,
            segment_type=self.segment_type,
            region_type=self.region_type,
            np_dtype=self.np_dtype,
        )


@dataclass
class AccountGroupInfo:
    """
    Represents a group of accounts with associated metadata and operations.

    This class manages multiple accounts as a single entity, organizing them
    in a 2D array where each row represents an account's time series data.
    It provides operations for accessing, transforming, and analyzing the
    entire group of accounts.

    Attributes
    ----------
    arr : ArrayF
        A 2D NumPy array containing data for all accounts, where each row
        represents one account's time series.
    account_map : dict[AccountType, int]
        A mapping of account types to their corresponding row indices in `arr`.
    dates : Sequence[dict[datetime.date, int]]
        A sequence of date dictionaries, one per account, mapping dates to
        column indices in `arr`.
    segment_type : HierarchyTree[ProductType]
        The product hierarchy level associated with all accounts in the group.
    region_type : HierarchyTree[LocationType]
        The location hierarchy level associated with all accounts in the group.
    np_dtype : type, default=BASE_NP_DTYPE
        The NumPy data type used for the account data.

    Methods
    -------
    from_account_lst(account_lst: list[AccountInfo]) -> AccountGroupInfo
        Creates an AccountGroupInfo from a list of AccountInfo objects.
    __getitem__(key: AccountType) -> AccountInfo
        Retrieves the AccountInfo object for the given account type.
    __len__() -> int
        Returns the number of accounts in the group.
    flip_account_map() -> dict[int, AccountType]
        Returns a reversed mapping of indices to account types.
    get_ordered_accounts() -> list[AccountType]
        Returns a list of account types ordered by their indices.
    flip_dates() -> list[dict[int, datetime.date]]
        Returns reversed date mappings for all accounts.
    apply_lag(max_lag: int) -> AccountGroupInfo
        Applies a lag to all accounts and returns a new instance.
    apply_daterange(start_date: datetime.date, end_date: datetime.date) -> AccountGroupInfo
        Filters all accounts to a specific date range.

    Notes
    -----
    The array data is automatically cast to the specified np_dtype in __post_init__.
    All accounts in the group share the same segment_type and region_type.
    Methods that transform the data return new AccountGroupInfo instances,
    preserving immutability of the original object.

    Examples
    --------
    Access a specific account from the group:

    >>> account_group = AccountGroupInfo(...)
    >>> revenue_account = account_group[AccountType('REVENUE')]

    Get all account types in order:

    >>> ordered_accounts = account_group.get_ordered_accounts()
    """

    arr: ArrayF
    account_map: dict[AccountType, int]
    dates: Sequence[dict[datetime.date, int]]
    segment_type: HierarchyTree[ProductType]
    region_type: HierarchyTree[LocationType]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        """Cast array to specified numpy dtype upon initialization."""
        # Ensure all account data uses consistent dtype
        self.arr = self.arr.astype(self.np_dtype)
        return

    @staticmethod
    def from_account_lst(account_lst: list[AccountInfo]) -> AccountGroupInfo:
        """
        Construct an AccountGroupInfo from a list of individual AccountInfo objects.

        This factory method combines multiple AccountInfo objects into a single
        AccountGroupInfo, creating a 2D array structure where each row represents
        one account's time series.

        Parameters
        ----------
        account_lst : list[AccountInfo]
            List of AccountInfo objects to combine. All objects must have:
            - Arrays of identical length (same number of time periods)
            - Compatible segment_type and region_type

        Returns
        -------
        AccountGroupInfo
            A new AccountGroupInfo containing all accounts from the input list.
            The segment_type, region_type, and np_dtype are inherited from the
            first AccountInfo in the list.

        Raises
        ------
        AssertionError
            If AccountInfo objects have arrays of different lengths.

        Notes
        -----
        - Account data is stacked row-wise in the order provided in the list
        - Each account's date mapping is preserved independently
        - All arrays are cast to the dtype of the first account

        Examples
        --------
        >>> revenue = AccountInfo(arr=..., account_type='REVENUE', ...)
        >>> cogs = AccountInfo(arr=..., account_type='COGS', ...)
        >>> group = AccountGroupInfo.from_account_lst([revenue, cogs])
        >>> len(group)
        2
        """
        # Get the length of the first account's array as reference
        len_acc1 = account_lst[0].arr.shape[0]

        # Validate that all accounts have arrays of the same length
        for account in account_lst[1:]:
            assert account.arr.shape[0] == len_acc1, (
                'All AccountInfo objects must have the same length arr to create AccountGroupInfo.'
            )

        # Initialize 2D array to hold all account data (rows = accounts, cols = time periods)
        new_arr: ArrayF = np.zeros(
            (len(account_lst), len_acc1),
            dtype=account_lst[0].np_dtype,
        )

        # Initialize mapping from account types to row indices
        new_acc_map: dict[AccountType, int] = {}

        # Initialize list of date mappings (one dict per account)
        new_dates: list[dict[datetime.date, int]] = []

        # Populate the array, account map, and dates list
        for idx, account in enumerate(account_lst):
            # Copy account data to the corresponding row, casting to common dtype
            new_arr[idx] = account.arr.astype(account_lst[0].np_dtype)

            # Map account type to its row index
            new_acc_map[account.account_type] = idx

            # Store the account's date mapping
            new_dates.append(account.dates)

        # Create and return new AccountGroupInfo with combined data
        # Metadata (segment_type, region_type, np_dtype) inherited from first account
        return AccountGroupInfo(
            arr=new_arr,
            account_map=new_acc_map,
            dates=new_dates,
            segment_type=account_lst[0].segment_type,
            region_type=account_lst[0].region_type,
            np_dtype=account_lst[0].np_dtype,
        )

    def __getitem__(self, key: AccountType) -> AccountInfo:
        """
        Access an individual account from the group by account type.

        Parameters
        ----------
        key : AccountType
            The account type identifier to retrieve.

        Returns
        -------
        AccountInfo
            An AccountInfo object containing the data for the specified account,
            including its time series array, date mapping, and metadata.

        Raises
        ------
        AssertionError
            If the specified AccountType is not present in this group.

        Examples
        --------
        >>> revenue = account_group[AccountType('REVENUE')]
        >>> revenue.arr.shape
        (24,)  # 24 time periods for this account
        """
        assert key in self.account_map.keys(), (
            'Given AccountType not present in this AccountGroupInfo.'
        )
        return AccountInfo(
            arr=self.arr[self.account_map[key]],
            account_type=key,
            dates=self.dates[self.account_map[key]],
            segment_type=self.segment_type,
            region_type=self.region_type,
            np_dtype=self.np_dtype,
        )

    def __len__(self) -> int:
        """
        Return the number of accounts in this group.

        Returns
        -------
        int
            The total number of accounts in the account_map.

        Examples
        --------
        >>> len(account_group)
        5  # This group contains 5 accounts
        """
        # Count accounts by getting number of keys in the account map
        return len(self.account_map.keys())

    def flip_account_map(self) -> dict[int, AccountType]:
        """
        Create a reverse mapping from array row indices to account types.

        Returns
        -------
        dict[int, AccountType]
            A dictionary mapping row indices in the array to their corresponding
            AccountType identifiers. This is the inverse of the `account_map` attribute.

        See Also
        --------
        flip_dict : Utility function used to reverse the dictionary.
        get_ordered_accounts : Retrieves accounts in index order using this mapping.

        Examples
        --------
        >>> account_group.account_map
        {AccountType('REVENUE'): 0, AccountType('COGS'): 1}
        >>> account_group.flip_account_map()
        {0: AccountType('REVENUE'), 1: AccountType('COGS')}
        """
        # Reverse the account_map to enable index-to-account lookups
        return flip_dict(dictionary=self.account_map)

    def get_ordered_accounts(self) -> list[AccountType]:
        """
        Retrieve all account types ordered by their row indices.

        Returns
        -------
        list[AccountType]
            A list of AccountType identifiers in the order they appear as rows
            in the data array (i.e., sorted by their index values in account_map).

        Examples
        --------
        >>> account_group.get_ordered_accounts()
        [AccountType('REVENUE'), AccountType('COGS'), AccountType('OPEX')]
        """
        # Get index-to-account mapping and extract accounts in order
        flipped_accs = self.flip_account_map()
        return [flipped_accs[k] for k in range(len(self))]

    def flip_dates(self) -> list[dict[int, datetime.date]]:
        """
        Create reverse date mappings for all accounts in the group.

        Returns
        -------
        list[dict[int, datetime.date]]
            A list of dictionaries, one per account, mapping array column indices
            to their corresponding dates. The list order matches the account order
            in the array.

        See Also
        --------
        flip_seq_dicts : Utility function used to reverse the sequence of dictionaries.

        Notes
        -----
        Each dictionary in the returned list corresponds to one account and maps
        that account's array column indices to dates.
        """
        # Reverse all date dictionaries in the sequence for index-to-date lookups
        return flip_seq_dicts(seq_dicts=self.dates)

    def _lag_dates(self, max_lag: int) -> list[dict[datetime.date, int]]:
        """
        Generate date-to-index mappings for lagged data across all accounts.

        This private method creates new date mappings that account for the loss
        of observations at the beginning of the time series due to lagging.

        Parameters
        ----------
        max_lag : int
            The number of time periods to lag. Must be less than the total
            number of time periods in the array.

        Returns
        -------
        list[dict[datetime.date, int]]
            A list of dictionaries (one per account) mapping dates to their
            new indices in the lagged array. Each dictionary excludes the
            first (max_lag + 1) observations.

        Notes
        -----
        For each account, dates are remapped such that the date originally at
        position (k + max_lag) now maps to position k in the lagged series.
        This accounts for the loss of initial observations due to lagging.

        See Also
        --------
        apply_lag : Public method that uses this to create lagged AccountGroupInfo.
        """
        # Get reverse mapping from indices to dates for all accounts
        flipped_dates = self.flip_dates()

        # Initialize list to hold date-to-index mappings for each account
        new_dates: list[dict[datetime.date, int]] = []

        # Process each account in order
        for account in self.get_ordered_accounts():
            # Initialize empty dictionary for this account's date mapping
            new_dates.append({})

            # Map dates to new indices, skipping the first (max_lag + 1) observations
            # which are lost due to lagging
            for k in range(self.arr.shape[1] - (max_lag + 1)):
                # Date at position (k + max_lag) in original series maps to index k in lagged series
                new_dates[self.account_map[account]][
                    flipped_dates[self.account_map[account]][k + max_lag]
                ] = k

        return new_dates

    def apply_lag(self, max_lag: int) -> AccountGroupInfo:
        """
        Create a lagged version of the account group by removing initial time periods.

        This method generates a new AccountGroupInfo with data shifted by
        removing the first max_lag columns (time periods) from all accounts.

        Parameters
        ----------
        max_lag : int
            The number of time periods to remove from the beginning of each
            account's time series.

        Returns
        -------
        AccountGroupInfo
            A new AccountGroupInfo instance with:
            - arr: Array with first max_lag columns removed
            - dates: Date mappings adjusted for removed periods
            - All other metadata preserved from the original

        Notes
        -----
        This method is commonly used in time series modeling to create lagged
        features while maintaining alignment across all accounts.

        Examples
        --------
        >>> original_group.arr.shape
        (5, 100)  # 5 accounts, 100 time periods
        >>> lagged_group = original_group.apply_lag(12)
        >>> lagged_group.arr.shape
        (5, 88)  # Same 5 accounts, 88 time periods (100 - 12)
        """
        # Create new instance with sliced array (remove first max_lag columns)
        return AccountGroupInfo(
            arr=self.arr[:, max_lag:],  # Slice all rows, remove first max_lag columns
            account_map=self.account_map,  # Preserve account mapping
            dates=self._lag_dates(
                max_lag=max_lag
            ),  # Adjust date mappings for all accounts
            segment_type=self.segment_type,
            region_type=self.region_type,
            np_dtype=self.np_dtype,
        )

    def apply_daterange(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> AccountGroupInfo:
        """
        Filter all accounts to a specified date range.

        This method creates a new AccountGroupInfo containing only data within
        the specified date range for all accounts. Each account is filtered
        individually and the results are combined into a new group.

        Parameters
        ----------
        start_date : datetime.date
            The starting date for the filtered range (inclusive).
        end_date : datetime.date
            The ending date for the filtered range (inclusive).

        Returns
        -------
        AccountGroupInfo
            A new AccountGroupInfo instance containing the filtered data for all
            accounts within the specified date range. The returned object maintains
            the same account mapping, segment type, region type, and numpy dtype as
            the original.

        Notes
        -----
        The number of columns in the resulting array equals the number of
        months between start_date and end_date (inclusive). All accounts
        maintain their original row order.

        Examples
        --------
        >>> original_group.arr.shape
        (5, 100)  # 5 accounts, 100 months of data
        >>> filtered = original_group.apply_daterange(
        ...     datetime.date(2023, 1, 1),
        ...     datetime.date(2023, 12, 1)
        ... )
        >>> filtered.arr.shape
        (5, 12)  # Same 5 accounts, only 12 months
        """
        # Initialize list to hold date-to-index mappings for each account after filtering
        new_dates: list[dict[datetime.date, int]] = []

        # Calculate number of months in the date range and initialize array
        # Rows = number of accounts, Cols = number of months in range
        new_arr: ArrayF = np.zeros(
            (len(self), month_dif(start_date=start_date, end_date=end_date) + 1),
            dtype=self.np_dtype,
        )

        # Process each account in order
        for idx, account in enumerate(self.get_ordered_accounts()):
            # Apply date range filter to individual account
            acc_info = self[account].apply_daterange(
                start_date=start_date,
                end_date=end_date,
            )

            # Store the filtered account's date mapping
            new_dates.append(acc_info.dates)

            # Copy the filtered account's data to the corresponding row
            new_arr[idx] = acc_info.arr

        # Create and return new AccountGroupInfo with filtered data
        # Preserve original account mapping and metadata
        return AccountGroupInfo(
            arr=new_arr.astype(self.np_dtype),
            account_map=self.account_map,
            dates=new_dates,
            segment_type=self.segment_type,
            region_type=self.region_type,
            np_dtype=self.np_dtype,
        )

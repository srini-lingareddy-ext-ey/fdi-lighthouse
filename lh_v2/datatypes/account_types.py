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

    Attributes
    ----------
    arr : ArrayF
        A NumPy array containing the time series data for the account.
    dates : dict[datetime.date, int]
        A mapping of dates to their corresponding indices in the `arr`.
    account_type : AccountType
        The type of the account.
    segment_type : SegmentType
        The segment type associated with the account.
    region_type : RegionType
        The region type associated with the account.
    np_dtype : type, optional
        The NumPy data type used for the account data, default is `BASE_NP_DTYPE`.

    Methods
    -------
    get_info() -> tuple[AccountType, SegmentType, RegionType]
        Retrieves the account type, segment type, and region type.
    flip_dates() -> dict[int, datetime.date]
        Returns a reversed mapping of indices to dates.
    apply_lag(max_lag: int) -> AccountInfo
        Applies a lag to the account data and returns a new AccountInfo instance.
    apply_daterange(start_date: Optional[datetime.date], end_date: Optional[datetime.date]) -> AccountInfo
        Filters the account data to a specific date range.
    add_forecast_vals(forecast_dates: Sequence[datetime.date], forecast_values: ArrayF) -> AccountInfo
        Adds forecast values to the account data.
    """

    arr: ArrayF
    dates: dict[datetime.date, int]
    account_type: AccountType
    segment_type: HierarchyTree[ProductType]
    region_type: HierarchyTree[LocationType]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        self.arr = self.arr.astype(self.np_dtype)
        return

    def get_info(self) -> tuple[AccountType, ProductType, LocationType]:
        """
        Retrieves information about the account type, segment type, and region type.

        Returns
        -------
        tuple[AccountType, SegmentType, RegionType]
            A tuple containing the account type, segment type, and region type.
        """
        return self.account_type, self.segment_type.name, self.region_type.name

    def flip_dates(self) -> dict[int, datetime.date]:
        """
        Return a flipped dictionary with dates as values and integers as keys.

        Returns
        -------
        dict[int, datetime.date]
            A dictionary with integer keys and date values, where the original
            dates dictionary's keys and values are swapped.

        See Also
        --------
        flip_dict : Function used to flip the dictionary keys and values.
        """
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
        return AccountInfo(
            arr=self.arr[max_lag:],
            dates=self._lag_dates(max_lag=max_lag),
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

    Attributes
    ----------
    arr : ArrayF
        A NumPy array containing data for the accounts.
    account_map : dict[AccountType, int]
        A mapping of account types to their corresponding indices in the `arr`.
    dates : Sequence[dict[datetime.date, int]]
        A sequence of date dictionaries mapping datetime64 objects to indices for each account.
    segment_type : SegmentType
        The segment type associated with the accounts.
    region_type : RegionType
        The region type associated with the accounts.
    np_dtype : type, optional
        The NumPy data type used for the account data, default is `BASE_NP_DTYPE`.

    Methods
    -------
    __getitem__(key: AccountType) -> AccountInfo
        Retrieves the `AccountInfo` object for the given account type.
    __len__() -> int
        Returns the number of accounts in the group.
    flip_account_map() -> dict[int, AccountType]
        Returns a reversed mapping of indices to account types.
    get_ordered_accounts() -> list[AccountType]
        Returns a list of account types in the order of their indices.
    flip_dates() -> list[dict[int, datetime.date]]
        Returns a reversed mapping of indices to datetime64 objects for all accounts.
    apply_lag(max_lag: int) -> AccountGroupInfo
        Applies a lag to the account data and returns a new AccountGroupInfo instance.
    """

    arr: ArrayF
    account_map: dict[AccountType, int]
    dates: Sequence[dict[datetime.date, int]]
    segment_type: HierarchyTree[ProductType]
    region_type: HierarchyTree[LocationType]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        self.arr = self.arr.astype(self.np_dtype)
        return

    @staticmethod
    def from_account_lst(account_lst: list[AccountInfo]) -> AccountGroupInfo:
        """
        Create an AccountGroupInfo object from a list of AccountInfo objects.

        Parameters
        ----------
        account_lst : list[AccountInfo]
            A list of AccountInfo objects to combine into a single AccountGroupInfo.
            All AccountInfo objects must have arrays of the same length.

        Returns
        -------
        AccountGroupInfo
            A new AccountGroupInfo object containing the combined data from all
            input AccountInfo objects.

        Raises
        ------
        AssertionError
            If the AccountInfo objects have arrays of different lengths.

        Notes
        -----
        The method creates a 2D array where each row corresponds to an AccountInfo
        object from the input list. All arrays are cast to the dtype of the first
        AccountInfo object in the list. The segment_type, region_type, and np_dtype
        are inherited from the first AccountInfo object.
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
        Retrieve the AccountInfo object corresponding to the given AccountType key.

        Parameters
        ----------
        key : AccountType
            The account type key to look up in the account group.

        Returns
        -------
        AccountInfo
            The AccountInfo object associated with the given AccountType key.

        Raises
        ------
        AssertionError
            If the given AccountType key is not present in the account_map.
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

    def __len__(self):
        """
        Returns the number of accounts in the account map.

        This method is used to retrieve the total count of keys in the `account_map` attribute.

        Returns
        -------
        int
            The number of keys in the `account_map`.
        """
        return len(self.account_map.keys())

    def flip_account_map(self) -> dict[int, AccountType]:
        """
        Reverses the `account_map` dictionary, swapping keys and values.

        Returns
        -------
        dict[int, AccountTypes]
            A dictionary where the keys are the values of the original `account_map`
            and the values are the keys of the original `account_map`.

        Notes
        -----
        This method assumes that the values in `account_map` are unique and hashable,
        as they will become the keys in the returned dictionary.
        """
        return flip_dict(dictionary=self.account_map)

    def get_ordered_accounts(self) -> list[AccountType]:
        """
        Retrieves a list of account types in a specific ordered sequence.

        Returns
        -------
        list[AccountTypes]
            A list of `AccountTypes` objects ordered based on the mapping
            defined by `flip_account_map` and the range of the current instance's length.
        """
        flipped_accs = self.flip_account_map()
        return [flipped_accs[k] for k in range(len(self))]

    def flip_dates(self) -> list[dict[int, datetime.date]]:
        """
        Flip the sequence of date dictionaries.

        Returns
        -------
        list[dict[int, datetime.date]]
            A flipped version of the dates sequence where the order of dictionaries
            in the sequence is reversed or transformed according to flip_seq_dicts.

        See Also
        --------
        flip_seq_dicts : The underlying function used to flip the sequence.

        Notes
        -----
        This method delegates to flip_seq_dicts to perform the actual flipping
        operation on the dates attribute.
        """
        return flip_seq_dicts(seq_dicts=self.dates)

    def _lag_dates(self, max_lag: int) -> list[dict[datetime.date, int]]:
        """
        Create a mapping of lagged dates to their corresponding indices.

        This method generates a list of dictionaries that map dates to their lagged
        indices for each account. The lag is applied by shifting the date indices
        backward by max_lag positions.

        Parameters
        ----------
        max_lag : int
            The maximum number of time periods to lag. Must be less than the total
            number of time periods in the array.

        Returns
        -------
        list[dict[datetime.date, int]]
            A list of dictionaries, one per account, where each dictionary maps
            a date to its lagged index position. The length of the list equals
            the number of accounts, and each dictionary contains mappings for
            dates shifted by max_lag positions.

        Notes
        -----
        The method uses flipped dates from `flip_dates()` and iterates through
        all ordered accounts. For each account, it creates a mapping where dates
        are offset by max_lag positions, effectively creating a lagged time series
        mapping.
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
        Apply a lag to the account group data by removing the first max_lag time periods.

        Parameters
        ----------
        max_lag : int
            The number of time periods to remove from the beginning of the data array.

        Returns
        -------
        AccountGroupInfo
            A new AccountGroupInfo instance with the lagged data. The array will have
            max_lag fewer time periods, and the dates will be adjusted accordingly.

        Notes
        -----
        This method creates a new instance and does not modify the original object.
        The account_map, segment_type, region_type, and np_dtype are preserved from
        the original instance.
        """
        return AccountGroupInfo(
            arr=self.arr[:, max_lag:],
            account_map=self.account_map,
            dates=self._lag_dates(max_lag=max_lag),
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
        Apply a date range filter to all accounts in the group.

        This method creates a new AccountGroupInfo instance with data filtered to the
        specified date range. It iterates through all accounts, applies the date range
        to each, and consolidates the results into a new array structure.

        Parameters
        ----------
        start_date : datetime.date
            The start date of the range to apply (inclusive).
        end_date : datetime.date
            The end date of the range to apply (inclusive).

        Returns
        -------
        AccountGroupInfo
            A new AccountGroupInfo instance containing the filtered data for all
            accounts within the specified date range. The returned object maintains
            the same account mapping, segment type, region type, and numpy dtype as
            the original.

        Notes
        -----
        The method preserves the order of accounts as returned by
        `get_ordered_accounts()` and creates a new numpy array with dimensions
        based on the number of accounts and the number of months in the date range.
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

import datetime
from dataclasses import dataclass
from typing import Any, NewType, Optional, Sequence, TypeVar, overload

import numpy as np
from dateutil.relativedelta import relativedelta

from lh_v2.shared import BASE_NP_DTYPE, ArrayF, ArrayI
from lh_v2.util import flip_dict, flip_seq_dicts, month_dif

from .account_types import AccountType

DriverName = NewType('DriverName', str)
DriverClassification = NewType('DriverClassification', str)

ClassificationOfDriver = TypeVar(
    'ClassificationOfDriver', AccountType, DriverClassification
)


@dataclass
class Driver:
    """
    A time series data container for driver variables with date indexing.

    The Driver class stores array data with associated dates, providing methods
    for date range filtering and forecast value addition. It maintains a mapping
    between dates and array indices for efficient time-based operations.

    Attributes
    ----------
    name : DriverName
        The name identifier for this driver.
    arr : ArrayF
        The numpy array containing the driver's time series values.
    dates : dict[datetime.date, int]
        A mapping from dates to their corresponding indices in the array.
    np_dtype : type, optional
        The numpy data type for the array, defaults to BASE_NP_DTYPE.

    Methods
    -------
    apply_daterange(start_date=None, end_date=None)
        Returns a new Driver instance filtered to the specified date range.
    add_forecast_vals(forecast_dates, forecast_values)
        Returns a new Driver instance with forecast values appended to historical data.

    Examples
    --------
    >>> driver = Driver(
    ...     name="temperature",
    ...     arr=np.array([20.0, 21.0, 22.0]),
    ...     dates={date(2023, 1, 1): 0, date(2023, 1, 2): 1, date(2023, 1, 3): 2}
    ... )
    >>> filtered = driver.apply_daterange(start_date=date(2023, 1, 2))
    >>> filtered.arr.shape[0]
    2
    """

    name: DriverName
    arr: ArrayF
    dates: dict[datetime.date, int]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        """Cast array to specified numpy dtype upon initialization."""
        self.arr = self.arr.astype(self.np_dtype)
        return

    def _apply_daterange_asserts(
        self,
        end_date: datetime.date,
        start_date: Optional[datetime.date] = None,
    ) -> None:
        """
        Validate that the specified date range is within available dates.

        This private method checks whether the provided start and end dates
        exist in the driver's date mapping, raising errors if they don't.

        Parameters
        ----------
        end_date : datetime.date
            The ending date to validate (required).
        start_date : Optional[datetime.date], default=None
            The starting date to validate. If None, validation is skipped
            for the start date.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If end_date is not present in the driver's available dates.
        ValueError
            If start_date is provided but not present in the driver's
            available dates.
        """
        if end_date not in self.dates.keys():
            raise ValueError(
                f'End date for driver {self.name} is not in available dates.'
            )
        if start_date is not None:
            if start_date not in self.dates.keys():
                raise ValueError(
                    f'Start date for driver {self.name} is not in available dates.'
                )
        return

    def apply_daterange(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> Driver:
        """
        Filter the driver data to a specified date range.

        Returns a new Driver instance containing only the data within the specified
        date range. If neither start_date nor end_date is provided, returns self unchanged.

        Parameters
        ----------
        start_date : datetime.date, optional
            The starting date of the range (inclusive). If None, starts from the
            beginning of the data.
        end_date : datetime.date, optional
            The ending date of the range (inclusive). If None, includes data up to
            the end of the array.

        Returns
        -------
        Driver
            A new Driver instance with filtered data and reindexed dates starting from 0.

        Examples
        --------
        >>> driver = Driver(
        ...     name="temperature",
        ...     arr=np.array([20.0, 21.0, 22.0, 23.0]),
        ...     dates={date(2023, 1, 1): 0, date(2023, 1, 2): 1,
        ...            date(2023, 1, 3): 2, date(2023, 1, 4): 3}
        ... )
        >>> filtered = driver.apply_daterange(start_date=date(2023, 1, 2),
        ...                                     end_date=date(2023, 1, 3))
        >>> filtered.arr
        array([21., 22.])
        >>> filtered.dates
        {date(2023, 1, 2): 0, date(2023, 1, 3): 1}
        """
        # If no date range specified, return the original driver unchanged
        if start_date is None and end_date is None:
            return self

        # Determine the ending index for array slicing
        if end_date is None:
            # If no end_date provided, use the full array length
            end_idx = self.arr.shape[0]
        else:
            # Add 1 to include the end_date in the range (inclusive)
            end_idx = self.dates[end_date] + 1

        # Determine the starting index for array slicing
        if start_date is None:
            # If no start_date provided, start from the beginning
            start_idx = 0
        else:
            # Get the index corresponding to the start_date
            start_idx = self.dates[start_date]

        # Create a new dates dictionary with reindexed values starting from 0
        # Only include dates that fall within the specified range
        new_dates: dict[datetime.date, int] = {
            date: idx - start_idx
            for date, idx in self.dates.items()
            if start_idx <= idx < end_idx
        }

        # Return a new Driver instance with the filtered array and reindexed dates
        return Driver(
            name=self.name,
            arr=self.arr[start_idx:end_idx].astype(self.np_dtype),
            dates=new_dates,
            np_dtype=self.np_dtype,
        )

    def add_forecast_vals(
        self,
        forecast_dates: Sequence[datetime.date],
        forecast_values: ArrayF,
    ) -> Driver:
        """
        Add forecast values to the driver, creating a new Driver instance.

        This method combines historical data (up to the first forecast date) with new
        forecast values, creating a continuous series from historical data through the
        forecast period.

        Parameters
        ----------
        forecast_dates : Sequence[datetime.date]
            Sequence of dates corresponding to the forecast period. Must be in
            chronological order and start after the last historical date.
        forecast_values : ArrayF
            Array of forecast values. Must have the same length as forecast_dates.
            The dtype will be converted to match the driver's np_dtype.

        Returns
        -------
        Driver
            A new Driver instance containing historical data up to the first forecast
            date, followed by the provided forecast values.

        Raises
        ------
        AssertionError
            If the length of forecast_dates does not match the length of forecast_values.

        Notes
        -----
        - Historical data after the first forecast date is discarded.
        - The returned Driver maintains the same name and dtype as the original.
        - Date indices are reassigned sequentially starting from 0.

        Examples
        --------
        >>> driver = Driver(name="revenue", arr=np.array([100, 110, 120]), ...)
        >>> forecast_dates = [date(2024, 1, 1), date(2024, 2, 1)]
        >>> forecast_values = np.array([130, 140])
        >>> new_driver = driver.add_forecast_vals(forecast_dates, forecast_values)
        """
        # Validate that the number of forecast dates matches the number of forecast values
        assert len(forecast_dates) == forecast_values.shape[0], (
            'Length of forecast_dates must match the number of forecast_values'
        )

        # Create new dates dictionary containing only historical dates before the first forecast date
        new_dates: dict[datetime.date, int] = {
            date: idx for date, idx in self.dates.items() if date < forecast_dates[0]
        }

        # Calculate the number of historical data points to retain
        n_hist = len(new_dates)

        # Find the last historical date (needed for extracting historical data)
        last_hist_date = max(new_dates.keys())

        # Add forecast dates to the dates dictionary with sequential indices after historical data
        for idx, date in enumerate(forecast_dates):
            new_dates[date] = n_hist + idx

        # Initialize new array to hold both historical and forecast values
        new_arr = np.zeros(
            (len(new_dates),),
            dtype=self.np_dtype,
        )

        # Copy historical data up to the last historical date into the new array
        new_arr[:n_hist] = self.apply_daterange(end_date=last_hist_date).arr.astype(
            self.np_dtype
        )

        # Append forecast values to the array after the historical data
        new_arr[n_hist:] = forecast_values.astype(self.np_dtype)

        # Return a new Driver instance with combined historical and forecast data
        return Driver(
            name=self.name,
            arr=new_arr,
            dates=new_dates,
            np_dtype=self.np_dtype,
        )


@dataclass
class DriverGroup:
    """
    Represents a group of drivers with associated data and mappings.

    Attributes
    ----------
    arr : ArrayF
        A NumPy array containing the driver data.
    map : dict[DriverName, int]
        A dictionary mapping driver names to their corresponding indices in the array.
    dates : list[dict[datetime.date, int]]
        A list of dictionaries mapping datetime64 objects to integer indices for each driver.
    np_dtype : type, default=BASE_NP_DTYPE
        The NumPy data type to which the array should be cast.

    Methods
    -------
    __post_init__()
        Ensures the array is cast to the specified NumPy data type.
    __getitem__(key)
        Retrieves elements from the array based on the provided key.
    __len__()
        Returns the number of drivers in the group.
    __radd__(other)
        Concatenates two DriverGroup objects along the first axis.
    __add__(other)
        Alias for `__radd__`.
    order(driver_ordering)
        Reorders the drivers according to the specified ordering.
    flip_map()
        Returns a dictionary with the mapping reversed (indices to driver names).
    get_ordered_drivers()
        Returns a list of driver names in index order.
    flip_dates()
        Returns the dates with keys and values flipped.
    apply_lags(best_lags, max_lag)
        Applies specified lags to the driver data.

    Examples
    --------
    >>> arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> map = {'driver1': 0, 'driver2': 1}
    >>> dates = [{datetime.date('2020-01-01'): 0}, {datetime.date('2020-01-01'): 0}]
    >>> dg = DriverGroup(arr=arr, map=map, dates=dates)
    >>> dg['driver1']
    array([1., 2., 3.])
    """

    arr: ArrayF
    map: dict[DriverName, int]
    dates: list[dict[datetime.date, int]]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        """Cast array to specified numpy dtype upon initialization."""
        self.arr = self.arr.astype(self.np_dtype)
        return

    @staticmethod
    def from_driver_lst(driver_lst: list[Driver], np_dtype: type = BASE_NP_DTYPE):
        """
        Create a DriverGroup from a list of Driver objects.

        Parameters
        ----------
        driver_lst : list[Driver]
            A list of Driver objects to combine into a DriverGroup. All drivers
            must have arrays of the same length.
        np_dtype : type, optional
            The numpy data type to use for the combined array. Defaults to BASE_NP_DTYPE.

        Returns
        -------
        DriverGroup
            A new DriverGroup instance containing all drivers from the input list.

        Raises
        ------
        AssertionError
            If the drivers in driver_lst have arrays of different lengths.

        Notes
        -----
        This method stacks the arrays from each driver vertically and creates
        a mapping from driver names to their corresponding row indices in the
        combined array. Each driver's date dictionary is preserved in a list.
        """
        # Get the length of the first driver's array to use as reference
        len_driver1 = driver_lst[0].arr.shape[0]

        # Validate that all drivers have arrays of the same length
        for driver in driver_lst:
            assert driver.arr.shape[0] == len_driver1, (
                'All drivers must have the same length arrays to create DriverGroup.'
            )

        # Initialize containers for the combined DriverGroup data
        new_map: dict[DriverName, int] = {}
        # Create a 2D array with shape (number of drivers, length of each driver array)
        new_arr: ArrayF = np.zeros(
            (len(driver_lst), driver_lst[0].arr.shape[0]), np_dtype
        )
        new_dates: list[dict[datetime.date, int]] = []

        # Iterate through each driver to populate the combined structures
        for idx, driver in enumerate(driver_lst):
            # Map each driver name to its corresponding row index in the array
            new_map[driver.name] = idx
            # Copy the driver's array data into the corresponding row
            new_arr[idx] = driver.arr.astype(np_dtype)
            # Preserve the driver's date-to-index mapping
            new_dates.append(driver.dates)

        # Return a new DriverGroup with the combined data
        return DriverGroup(arr=new_arr, map=new_map, dates=new_dates, np_dtype=np_dtype)

    @overload
    def __getitem__(self, key: DriverName) -> ArrayF: ...
    @overload
    def __getitem__(self, key: int) -> ArrayF: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayF: ...
    @overload
    def __getitem__(self, key: tuple[slice, int]) -> ArrayF: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> ArrayF: ...
    @overload
    def __getitem__(self, key: tuple[int, slice]) -> ArrayF: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> float: ...
    @overload
    def __getitem__(self, key: ArrayI) -> ArrayF: ...
    def __getitem__(self, key: Any):
        """
        Retrieve an item or a subset of items from the object based on the provided key.

        Parameters
        ----------
        key : str, int, slice, tuple, or np.ndarray
            The key used to access the data. Accepted types include:
            - str: A string key mapped to an index.
            - int: An integer index.
            - slice: A slice object for slicing the data.
            - tuple: A tuple of two elements, where each element can be an int or slice.
            - np.ndarray: A NumPy array used for advanced indexing.

        Returns
        -------
        object
            The retrieved item(s) from the data structure. The return type depends on the key:
            - For `str`, `int`, or `slice`, the corresponding item(s) from the array.
            - For a tuple of two elements, the corresponding item(s) from a 2D array.
            - For `np.ndarray`, the result of advanced indexing.

        Raises
        ------
        ValueError
            If the provided key is not of an accepted type or format.
        """
        # Use pattern matching to handle different key types and formats
        match key:
            case str():
                # When key is a string, look up its index in the map and return corresponding row
                key = DriverName(key)
                return self.arr[self.map[key]]
            case int():
                # When key is an integer, return the corresponding row
                return self.arr[key]
            case slice():
                # When key is a slice, return the corresponding rows
                return self.arr[key]
            case int(), int() if len(key) == 2:
                # When key is a tuple of two integers, return a single element as float
                return float(self.arr[key[0], key[1]])
            case slice(), int() if len(key) == 2:
                # When key is a tuple with slice and integer, return selected rows from a specific column
                return self.arr[key[0], key[1]]
            case int(), slice() if len(key) == 2:
                # When key is a tuple with integer and slice, return selected columns from a specific row
                return self.arr[key[0], key[1]]
            case slice(), slice() if len(key) == 2:
                # When key is a tuple of two slices, return the corresponding subarray
                return self.arr[key[0], key[1]]
            case np.ndarray():
                # When key is a NumPy array, use it for advanced indexing
                return self.arr[key]
            case _:
                # Default case: unsupported key type
                print(type(key))
                raise ValueError('Key is not accepted...')

    def get_driver(self, driver_name: DriverName) -> Driver:
        """
        Extract a single Driver object from the group by name.

        Parameters
        ----------
        driver_name : DriverName
            The name of the driver to retrieve.

        Returns
        -------
        Driver
            A Driver object containing the data, dates, and metadata for
            the specified driver.

        Raises
        ------
        AssertionError
            If the driver_name is not present in the DriverGroup's map.

        Examples
        --------
        >>> driver_group = DriverGroup(...)
        >>> temp_driver = driver_group.get_driver(DriverName('temperature'))
        """
        assert driver_name in self.map, (
            f'Driver name {driver_name} not found in DriverGroup.'
        )
        return Driver(
            name=driver_name,
            arr=self.arr[self.map[driver_name]].astype(self.np_dtype),
            dates=self.dates[self.map[driver_name]],
            np_dtype=self.np_dtype,
        )

    def __len__(self) -> int:
        """
        Return the number of drivers in this group.

        Returns
        -------
        int
            The number of drivers (rows) in the driver array.
        """
        return self.arr.shape[0]

    def __radd__(self, other: Any) -> DriverGroup:
        """
        Implements the reverse addition operation for the DriverGroup class.

        This method allows the addition of a DriverGroup instance to another
        DriverGroup instance using the `+` operator, where the current instance
        is on the right-hand side of the operation.

        Parameters
        ----------
        other : DriverGroup
            The DriverGroup instance to be added to the current instance.

        Returns
        -------
        DriverGroup
            A new DriverGroup instance containing the concatenated arrays and
            updated driver mappings from both instances.

        Raises
        ------
        ValueError
            If the `other` parameter is not an instance of DriverGroup.

        Notes
        -----
        - The `arr` attributes of both DriverGroup instances are concatenated
          along the first axis.
        - The `map` dictionaries are merged, with indices from the `other`
          instance adjusted to account for the size of the current instance's
          array.
        """
        if isinstance(other, DriverGroup):
            # Concatenate arrays from both DriverGroup instances along axis 0 (rows)
            arr_drivers = np.concatenate(
                (self.arr, other.arr), axis=0, dtype=self.np_dtype
            )

            # Create new dictionary for merged driver mappings
            dict_drivers = {}

            # Copy mappings from current instance directly
            for driver, ind in self.map.items():
                dict_drivers[driver] = ind

            # Copy mappings from other instance, adjusting indices by adding current array's row count
            for driver, ind in other.map.items():
                dict_drivers[driver] = ind + self.arr.shape[0]

            # Create new list for dates
            new_dates = self.dates + other.dates

            # Return a new DriverGroup with the concatenated array and merged mappings
            return DriverGroup(
                arr=arr_drivers,
                map=dict_drivers,
                dates=new_dates,
                np_dtype=self.np_dtype,
            )

        # Raise error if other is not a DriverGroup instance
        raise ValueError('This type input is not supported.')

    def __add__(self, other: Any) -> DriverGroup:
        return self.__radd__(other)

    def order(self, driver_ordering: list[DriverName]) -> DriverGroup:
        """
        Reorder the drivers in the DriverGroup according to a specified ordering.

        Parameters
        ----------
        driver_ordering : list[DriverName]
            A list of driver names specifying the desired order of drivers.
            Must contain the same drivers as currently in the DriverGroup.

        Returns
        -------
        DriverGroup
            A new DriverGroup instance with drivers reordered according to
            driver_ordering. The array data, mapping, and dates are all
            reordered to match the new driver order.

        Notes
        -----
        This method creates a new DriverGroup rather than modifying the
        existing one in place.
        """
        # Create a new mapping dictionary where each driver name maps to its new index
        # based on its position in the driver_ordering list
        new_map: dict[DriverName, int] = {
            driver: idx for idx, driver in enumerate(driver_ordering)
        }

        # Initialize a new array with the same shape as the original
        # This will hold the reordered driver data
        new_arr: ArrayF = np.zeros(self.arr.shape, self.np_dtype)

        # Iterate through the driver ordering and copy each driver's data
        # from its old position to its new position in the array
        for idx, driver in enumerate(driver_ordering):
            new_arr[idx] = self.arr[self.map[driver]]

        # Create a new dates list, reordering the date dictionaries to match
        # the new driver order specified in driver_ordering
        new_dates: list[dict[datetime.date, int]] = [
            self.dates[self.map[driver]] for driver in driver_ordering
        ]

        # Return a new DriverGroup with the reordered data, maintaining the same dtype
        return DriverGroup(
            arr=new_arr,
            map=new_map,
            dates=new_dates,
            np_dtype=self.np_dtype,
        )

    def flip_map(self) -> dict[int, DriverName]:
        """
        Create a reverse mapping from indices to driver names.

        Returns
        -------
        dict[int, DriverName]
            A dictionary mapping driver indices to their corresponding
            driver names. This is the inverse of the `map` attribute.

        See Also
        --------
        flip_dict : Utility function used to reverse the dictionary.
        """
        return flip_dict(dictionary=self.map)

    def get_ordered_drivers(self) -> list[DriverName]:
        """
        Get an ordered list of driver names based on their positions.

        Returns
        -------
        list[DriverName]
            A list of driver names ordered by their position indices from 0 to len(self)-1.

        Notes
        -----
        This method first flips the internal mapping using flip_map() to get a position-to-driver
        mapping, then constructs an ordered list by iterating through positions sequentially.
        """
        flipped_drivers = self.flip_map()
        return [flipped_drivers[k] for k in range(len(self))]

    def flip_dates(self) -> list[dict[int, datetime.date]]:
        """
        Flip the sequence of date dictionaries.

        Returns
        -------
        list[dict[int, datetime.date]]
            A list of dictionaries with flipped key-value pairs from the original dates.
            Each dictionary in the sequence has its integer keys and datetime.date values
            transposed using the flip_seq_dicts utility function.

        See Also
        --------
        flip_seq_dicts : Utility function that performs the flipping operation.
        """
        return flip_seq_dicts(seq_dicts=self.dates)

    def _lag_driver_dates(
        self,
        flipped_dates: dict[int, datetime.date],
        driver_lag: int,
        max_lag: int,
    ) -> dict[datetime.date, int]:
        """
        Create a mapping of lagged dates to column indices for a driver variable.

        This method adjusts date indices based on the driver's specific lag relative to
        a maximum lag value, accounting for the temporal offset between different driver
        variables in the analysis.

        Parameters
        ----------
        flipped_dates : dict[int, datetime.date]
            A dictionary mapping column indices to their corresponding dates.
        driver_lag : int
            The specific lag (in time periods) associated with this driver variable.
        max_lag : int
            The maximum lag value across all driver variables in the system.

        Returns
        -------
        dict[datetime.date, int]
            A dictionary mapping adjusted dates to their corresponding column indices,
            accounting for the driver-specific lag offset.

        Notes
        -----
        The method iterates through valid column indices (excluding the last `max_lag`
        columns) and creates new date-to-index mappings by offsetting each date by
        the difference between `max_lag` and `driver_lag`.
        """
        # Initialize a new dictionary to store the date-to-index mapping after lag adjustment
        new_dates: dict[datetime.date, int] = {}

        # Iterate through valid column indices after accounting for max_lag truncation
        # The range is limited to (shape[1] - max_lag) to ensure we don't access out-of-bounds dates
        for k in range(self.arr.shape[1] - max_lag):
            # Map each output column index k to its corresponding date after lag adjustment
            # The date is fetched from the original flipped_dates by offsetting the index:
            # - Add max_lag to account for the overall truncation of the array
            # - Subtract driver_lag to apply this driver's specific lag offset
            # This effectively shifts the date by (max_lag - driver_lag) periods
            new_dates[flipped_dates[k + max_lag - driver_lag]] = k

        return new_dates

    def _lag_dates(
        self, best_lags: dict[DriverName, int], max_lag: int
    ) -> list[dict[datetime.date, int]]:
        """
        Lag the dates for each driver according to their best lag values.

        This method processes each driver's dates by applying the corresponding lag
        transformation, creating a list of date dictionaries where each date is mapped
        to its lag value.

        Parameters
        ----------
        best_lags : dict[DriverName, int]
            A dictionary mapping each driver name to its optimal lag value (in days).
        max_lag : int
            The maximum lag value allowed for any driver.

        Returns
        -------
        list[dict[datetime.date, int]]
            A list of dictionaries, one per driver in order, where each dictionary
            maps dates to their corresponding lag values after transformation.

        Notes
        -----
        The method first flips the dates using `flip_dates()`, then processes each
        driver in the order returned by `get_ordered_drivers()`. For each driver,
        it applies the lag transformation using `_lag_driver_dates()`.
        """
        # Flip the dates dictionary structure from {date: idx} to {idx: date} for each driver
        # This makes it easier to look up dates by column index when applying lags
        flipped_dates = self.flip_dates()

        # Initialize list to store the new date mappings for each driver after lag adjustment
        new_dates: list[dict[datetime.date, int]] = []

        # Iterate through each driver in the order they appear in the DriverGroup
        for driver in self.get_ordered_drivers():
            # Apply lag transformation to this driver's dates
            # - Use the driver's position to get its flipped dates from the list
            # - Apply the driver's specific lag value from best_lags
            # - Use max_lag as reference point for alignment across all drivers
            new_dates.append(
                self._lag_driver_dates(
                    flipped_dates=flipped_dates[self.map[driver]],
                    driver_lag=best_lags[driver],
                    max_lag=max_lag,
                )
            )

        return new_dates

    def apply_lags(self, best_lags: dict[DriverName, int], max_lag: int) -> DriverGroup:
        """
        Apply time lags to drivers and align them temporally.

        This method shifts each driver's time series according to the specified lag values
        and truncates the array to maintain alignment across all drivers. The resulting
        array is shortened by max_lag to ensure all lagged drivers have valid data.

        Parameters
        ----------
        best_lags : dict[DriverName, int]
            Dictionary mapping each driver name to its optimal lag value (in time steps).
            Must contain all drivers present in this DriverGroup.
        max_lag : int
            The maximum lag value across all drivers. Used to determine the amount of
            truncation needed to align all time series.

        Returns
        -------
        DriverGroup
            A new DriverGroup instance with lagged driver data, truncated dates, and
            the same driver mapping and dtype as the original.

        Raises
        ------
        AssertionError
            If the keys in best_lags do not exactly match the drivers in this DriverGroup.

        Notes
        -----
        - For a driver with lag=0, the data is taken from max_lag onwards
        - For a driver with lag>0, the data is shifted backwards by that amount
        - The final array shape is (n_drivers, original_length - max_lag)
        - Dates are adjusted accordingly using the _lag_dates method

        Examples
        --------
        >>> best_lags = {'temperature': 0, 'humidity': 2}
        >>> max_lag = 2
        >>> lagged_group = driver_group.apply_lags(best_lags, max_lag)
        """
        # Validate that best_lags contains exactly the same drivers as this DriverGroup
        assert set(self.map.keys()) == set(best_lags.keys()), (
            'Given best lags does not contain the same drivers as Driver Group.'
        )

        # Initialize a new array with truncated length (original_length - max_lag)
        # This ensures all lagged drivers align to the same temporal window
        new_driver_arr = np.zeros(
            (self.arr.shape[0], (self.arr.shape[1] - max_lag)),
            dtype=self.np_dtype,
        )

        # Apply the appropriate lag transformation to each driver
        for driver in self.map.keys():
            # Case 1: No lag (lag=0) - take data starting from max_lag position onwards
            # This aligns with the most-lagged driver's earliest valid data point
            if best_lags[driver] == 0:
                new_driver_arr[self.map[driver]] = self.arr[self.map[driver], max_lag:]
            # Case 2: Positive lag - shift the driver backwards in time
            # Start from (max_lag - driver_lag) and end at (-driver_lag) to exclude the last lag periods
            # This effectively aligns the lagged driver with the current time period
            else:
                new_driver_arr[self.map[driver]] = self.arr[
                    self.map[driver],
                    max_lag - best_lags[driver] : -best_lags[driver],
                ]

        # Return a new DriverGroup with lagged data, adjusted dates, and preserved metadata
        return DriverGroup(
            arr=new_driver_arr,
            map=self.map,
            dates=self._lag_dates(best_lags=best_lags, max_lag=max_lag),
            np_dtype=self.np_dtype,
        )

    def _apply_daterange_asserts(
        self,
        start_dates: dict[DriverName, datetime.date],
        end_dates: dict[DriverName, datetime.date],
    ) -> tuple[dict[DriverName, tuple[int, int]], int]:
        """
        Validate and process date ranges for drivers, returning indices and length.

        This method ensures that the provided start and end dates are valid for all
        drivers in the group, and that all resulting date ranges have the same length.

        Parameters
        ----------
        start_dates : dict[DriverName, datetime.date]
            Dictionary mapping driver names to their respective start dates.
        end_dates : dict[DriverName, datetime.date]
            Dictionary mapping driver names to their respective end dates.

        Returns
        -------
        indices : dict[DriverName, tuple[int, int]]
            Dictionary mapping driver names to tuples of (start_index, end_index)
            representing the slice indices for the date range.
        length : int
            The length of the date range (same for all drivers).

        Raises
        ------
        AssertionError
            If the keys in start_dates or end_dates don't match the drivers in the group.
        ValueError
            If any start or end date is not in the available dates for its driver.
        ValueError
            If the date ranges result in different lengths across drivers.

        Notes
        -----
        The end_index is incremented by 1 to make the range inclusive of the end_date.
        """
        # Validate that start_dates contains exactly the same drivers as this DriverGroup
        assert set(self.map.keys()) == set(start_dates.keys()), (
            'Given start dates does not contain the same drivers as Driver Group.'
        )
        # Validate that end_dates contains exactly the same drivers as this DriverGroup
        assert set(self.map.keys()) == set(end_dates.keys()), (
            'Given end dates does not contain the same drivers as Driver Group.'
        )

        # Verify that all provided start and end dates are valid for their respective drivers
        for driver in self.map.keys():
            # Check if the start date exists in this driver's available dates
            if start_dates[driver] not in self.dates[self.map[driver]].keys():
                raise ValueError(
                    f'Start date for driver {driver} is not in available dates.'
                )
            # Check if the end date exists in this driver's available dates
            if end_dates[driver] not in self.dates[self.map[driver]].keys():
                raise ValueError(
                    f'End date for driver {driver} is not in available dates.'
                )

        # Initialize list to track the length of each driver's date range
        lengths: list[int] = []
        # Initialize dictionary to store the start and end indices for each driver
        indices: dict[DriverName, tuple[int, int]] = {}

        # Calculate start and end indices for each driver's date range
        for driver in self.map.keys():
            # Get the array index corresponding to the start date for this driver
            start_idx = self.dates[self.map[driver]][start_dates[driver]]
            # Get the array index corresponding to the end date, +1 to make the range inclusive
            end_idx = (
                self.dates[self.map[driver]][end_dates[driver]] + 1
            )  # +1 to include end_date
            # Record the length of this driver's date range
            lengths.append(end_idx - start_idx)
            # Store the (start, end) index tuple for this driver
            indices[driver] = (start_idx, end_idx)

        # Ensure all drivers have the same date range length for proper alignment
        if len(set(lengths)) != 1:
            raise ValueError(
                'The given date ranges result in different lengths for different drivers.'
            )

        # Return the indices dictionary and the common length across all drivers
        return indices, lengths[0]

    def apply_daterange(
        self,
        start_dates: dict[DriverName, datetime.date],
        end_dates: dict[DriverName, datetime.date],
    ) -> DriverGroup:
        """
        Apply date range filtering to the driver group data.

        This method creates a new DriverGroup instance containing only the data within
        the specified date ranges for each driver. It filters both the data array and
        the date mappings accordingly.

        Parameters
        ----------
        start_dates : dict[DriverName, datetime.date]
            Dictionary mapping driver names to their respective start dates (inclusive).
        end_dates : dict[DriverName, datetime.date]
            Dictionary mapping driver names to their respective end dates (exclusive).

        Returns
        -------
        DriverGroup
            A new DriverGroup instance containing the filtered data and date mappings
            for the specified date ranges.

        Notes
        -----
        - The method performs assertion checks via `_apply_daterange_asserts` to
          validate the input date ranges.
        - The original DriverGroup instance remains unchanged.
        - Date indices are remapped relative to the new start position for each driver.
        - The filtered array maintains the same driver ordering and dtype as the original.

        See Also
        --------
        _apply_daterange_asserts : Validation method for date range parameters.
        """
        # Validate date ranges and extract start/end indices for each driver
        # Also ensures all drivers have the same resulting length for alignment
        indices, length = self._apply_daterange_asserts(
            start_dates=start_dates, end_dates=end_dates
        )

        # Initialize a new array to hold the filtered data for all drivers
        # Shape is (number of drivers, common filtered length)
        new_driver_arr = np.zeros(
            (self.arr.shape[0], length),
            dtype=self.np_dtype,
        )

        # Iterate through drivers in their original order to maintain consistency
        for driver in self.get_ordered_drivers():
            # Extract the start and end indices for this driver's date range
            start_idx, end_idx = indices[driver]
            # Copy the filtered slice of data from the original array into the new array
            # The driver's position in the array is maintained using self.map[driver]
            new_driver_arr[self.map[driver]] = self.arr[
                self.map[driver], start_idx:end_idx
            ]

        # Create a new list of date mappings for each driver, one dictionary per driver
        new_dates: list[dict[datetime.date, int]] = []
        for driver in self.get_ordered_drivers():
            # For each driver, create a new date-to-index mapping
            # Remap indices relative to the new start position (subtract start_idx)
            # Only include dates that fall within the filtered range [start_idx, end_idx)
            new_dates.append(
                {
                    date: idx - indices[driver][0]
                    for date, idx in self.dates[self.map[driver]].items()
                    if indices[driver][0] <= idx < indices[driver][1]
                }
            )

        # Return a new DriverGroup with the filtered array, original mapping, and updated dates
        return DriverGroup(
            arr=new_driver_arr,
            map=self.map,
            dates=new_dates,
            np_dtype=self.np_dtype,
        )

    def apply_daterange_lags(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        lags: dict[DriverName, int],
        b_training: bool,
    ) -> DriverGroup:
        """
        Apply date range filters with lag adjustments to each driver in the group.

        This method adjusts the start and end dates for each driver based on specified
        lag values (in months) and applies the corresponding date range filters.

        Parameters
        ----------
        start_date : datetime.date
            The base start date before lag adjustments.
        end_date : datetime.date
            The base end date before lag adjustments.
        lags : dict[DriverName, int]
            A dictionary mapping driver names to their respective lag values in months.
            Must contain exactly the same driver names as in the DriverGroup.
        b_training : bool
            If True, adjusts the global start_date forward by the maximum lag value
            to ensure consistent training data alignment.

        Returns
        -------
        DriverGroup
            A new DriverGroup instance with date range filters applied according to
            the specified lags for each driver.

        Raises
        ------
        AssertionError
            If the set of drivers in `lags` does not match the set of drivers in
            the DriverGroup's map.

        Notes
        -----
        - Positive lag values shift dates backward (earlier dates).
        - When `b_training` is True, the start_date is first shifted forward by
          max(lags) months before individual driver lags are applied.
        - Each driver's date range is calculated as:
            - start_dates[driver] = start_date - relativedelta(months=lags[driver])
            - end_dates[driver] = end_date - relativedelta(months=lags[driver])
        """
        # Validate that lags contains exactly the same drivers as this DriverGroup
        assert set(self.map.keys()) == set(lags.keys()), (
            'Given lags does not contain the same drivers as Driver Group.'
        )

        # If in training mode, adjust the start_date forward by the maximum lag
        # This ensures all drivers have sufficient historical data for their respective lags
        if b_training:
            start_date = start_date + relativedelta(months=max(lags.values()))

        # Initialize dictionaries to store the lag-adjusted start and end dates for each driver
        start_dates: dict[DriverName, datetime.date] = {}
        end_dates: dict[DriverName, datetime.date] = {}

        # Calculate lag-adjusted date ranges for each driver
        # Subtracting the lag shifts the dates backward, effectively accessing earlier data
        for driver in lags.keys():
            # Shift the start date backward by the driver's lag value (in months)
            start_dates[driver] = start_date - relativedelta(months=lags[driver])
            # Shift the end date backward by the driver's lag value (in months)
            end_dates[driver] = end_date - relativedelta(months=lags[driver])

        # Apply the calculated date ranges to filter the driver data
        return self.apply_daterange(
            start_dates=start_dates,
            end_dates=end_dates,
        )


@dataclass
class ClassifiedDriverGroups[ClassificationOfDriver]:
    """
    Represents multiple driver groups organized by classification.

    This class manages driver data that is categorized into different classifications,
    where each classification contains a subset of drivers with their own mappings.

    Attributes
    ----------
    arr : ArrayF
        A NumPy array containing all driver data across classifications.
    classification_groups : dict[DriverClassification, int]
        A dictionary mapping classification names to their corresponding indices.
    maps : Sequence[dict[DriverName, int]]
        A sequence of dictionaries, one per classification, mapping driver names
        to their indices in the array.
    dates : list[dict[datetime.date, int]]
        A list of dictionaries mapping datetime64 objects to integer indices for each driver.
    np_dtype : type, default=BASE_NP_DTYPE
        The NumPy data type to which the array should be cast.

    Methods
    -------
    __post_init__()
        Ensures the array is cast to the specified NumPy data type.
    __getitem__(key)
        Retrieves a DriverGroup for the specified classification.
    combine()
        Combines all classifications into a single DriverGroup.
    get_ordered_classifications()
        Returns a list of classification names in index order.
    get_ordered_drivers(classification)
        Returns a list of driver names for a specific classification in index order.
    get_all_ordered_drivers()
        Returns a list of all driver names across classifications in index order.
    flip_dates()
        Returns the dates with keys and values flipped.
    apply_lag(best_lags, max_lag)
        Applies specified lags to the driver data for each classification.

    Examples
    --------
    >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> classification_groups = {'type_a': 0, 'type_b': 1}
    >>> maps = [{'driver1': 0}, {'driver2': 1}]
    >>> dates = [{datetime.date('2020-01-01'): 0}, {datetime.date('2020-01-01'): 0}]
    >>> cdg = ClassifiedDriverGroups(arr=arr, classification_groups=classification_groups,
    ...                               maps=maps, dates=dates)
    >>> dg = cdg['type_a']
    >>> dg.arr
    array([[1., 2.]])
    """

    arr: ArrayF
    classification_groups: dict[ClassificationOfDriver, int]
    maps: Sequence[dict[DriverName, int]]
    dates: list[dict[datetime.date, int]]
    np_dtype: type = BASE_NP_DTYPE

    def __post_init__(self):
        """Cast array to specified numpy dtype upon initialization."""
        self.arr = self.arr.astype(self.np_dtype)
        return

    @staticmethod
    def from_driver_group_lst(
        driver_group_lst: list[DriverGroup],
        classification_groups: dict[ClassificationOfDriver, int],
    ) -> ClassifiedDriverGroups[ClassificationOfDriver]:
        """
        Create a ClassifiedDriverGroups from a list of DriverGroup objects.

        This factory method combines multiple DriverGroup objects into a single
        ClassifiedDriverGroups, where each DriverGroup represents a distinct
        classification category.

        Parameters
        ----------
        driver_group_lst : list[DriverGroup]
            List of DriverGroup objects to combine. All groups must have arrays
            with the same number of columns (time periods).
        classification_groups : dict[ClassificationOfDriver, int]
            Dictionary mapping classification names to their corresponding
            indices in the driver_group_lst.

        Returns
        -------
        ClassifiedDriverGroups[ClassificationOfDriver]
            A new ClassifiedDriverGroups instance containing all drivers from
            the input list, organized by their classifications.

        Raises
        ------
        AssertionError
            If the length of driver_group_lst doesn't match the length of
            classification_groups.
        AssertionError
            If the DriverGroup objects have arrays with different numbers
            of columns.

        Notes
        -----
        - Drivers are stacked vertically in the order they appear in the list
        - Each classification maintains its own driver-to-index mapping
        - All arrays are cast to the dtype of the first DriverGroup
        """
        assert len(driver_group_lst) == len(classification_groups), (
            'Length of driver_group_lst must match length of classification_groups.'
        )
        len_arr = driver_group_lst[0].arr.shape[1]
        for dg in driver_group_lst:
            assert dg.arr.shape[1] == len_arr, (
                'All DriverGroups must have the same number of columns to be combined.'
            )
        n_drivers = sum([len(dg) for dg in driver_group_lst])
        combined_arr = np.zeros(
            (n_drivers, len_arr), dtype=driver_group_lst[0].np_dtype
        )
        combined_maps: list[dict[DriverName, int]] = []
        combined_dates: list[dict[datetime.date, int]] = []

        c_idx = 0
        for idx in range(len(driver_group_lst)):
            class_map: dict[DriverName, int] = {}
            for driver in driver_group_lst[idx].get_ordered_drivers():
                combined_arr[c_idx] = driver_group_lst[idx][driver]
                class_map[driver] = c_idx
                combined_dates.append(
                    driver_group_lst[idx].dates[driver_group_lst[idx].map[driver]]
                )
                c_idx += 1
            combined_maps.append(class_map)

        return ClassifiedDriverGroups(
            arr=combined_arr,
            classification_groups=classification_groups,
            maps=combined_maps,
            dates=combined_dates,
            np_dtype=driver_group_lst[0].np_dtype,
        )

    def __getitem__(self, key: ClassificationOfDriver) -> DriverGroup:
        """
        Retrieve a subset of drivers based on their classification.

        Parameters
        ----------
        key : ClassificationOfDriver
            The classification key used to filter and retrieve a specific group of drivers.

        Returns
        -------
        DriverGroup
            A new DriverGroup instance containing only the drivers that match the specified
            classification. The returned group includes:
            - A subset of the array data corresponding to the selected drivers
            - A remapped dictionary with new sequential indices
            - The corresponding dates for the selected drivers
            - The same numpy dtype as the original group

        Raises
        ------
        AssertionError
            If the provided key is not present in the classification_groups keys.

        Notes
        -----
        This method creates a new DriverGroup by:
        1. Looking up the classification group associated with the key
        2. Extracting the relevant drivers and their original indices
        3. Creating a new array with data only for those drivers
        4. Remapping indices to be sequential starting from 0
        5. Preserving the dates associated with each driver's original index
        """
        # Validate that the requested classification exists in the group
        assert key in self.classification_groups.keys()

        # Extract the list of driver names that belong to this classification
        # Uses the classification_groups mapping to find the right map, then gets all driver names
        drivers = list(self.maps[self.classification_groups[key]].keys())

        # Get the original array indices for each driver in this classification
        # These indices reference positions in the full ClassifiedDriverGroups array
        old_inds = [
            self.maps[self.classification_groups[key]][driver] for driver in drivers
        ]

        # Extract rows from the full array that correspond to this classification's drivers
        # This creates a new array containing only the data for the selected drivers
        new_arr = self.arr[old_inds]

        # Create a new driver mapping with sequential indices starting from 0
        # This remapping ensures the DriverGroup has a clean 0-indexed structure
        new_map = {drivers[idx]: idx for idx in range(len(drivers))}

        # Extract the date mappings for each driver using their original indices
        # Preserves the date-to-column mapping for each driver in the new group
        new_dates = [self.dates[idx] for idx in old_inds]

        # Return a new DriverGroup containing only the drivers from this classification
        return DriverGroup(
            arr=new_arr,
            map=new_map,
            dates=new_dates,
            np_dtype=self.np_dtype,
        )

    def combine(self) -> DriverGroup:
        """
        Combine multiple driver maps into a single DriverGroup.

        This method merges all driver mappings from the maps collection into a single
        unified mapping, potentially overwriting earlier index assignments if the same
        driver appears in multiple maps.

        Returns
        -------
        DriverGroup
            A new DriverGroup instance with a combined driver mapping that includes
            all drivers from the individual maps. The array, dates, and numpy dtype
            are preserved from the original object.

        Notes
        -----
        If a driver appears in multiple maps, the index from the last map in the
        iteration order will be used in the final mapping.
        """
        # Initialize an empty dictionary to hold the combined driver-to-index mapping
        new_map: dict[DriverName, int] = {}

        # Iterate through each classification's driver map
        for old_map in self.maps:
            # Merge each driver and its array index into the combined mapping
            # Note: If a driver appears in multiple classifications, the last
            # occurrence will overwrite previous index assignments
            for driver, idx in old_map.items():
                new_map[driver] = idx

        # Return a new DriverGroup with all drivers combined into a single map
        # The array, dates, and dtype are preserved unchanged from the original
        return DriverGroup(
            arr=self.arr,
            map=new_map,
            dates=self.dates,
            np_dtype=self.np_dtype,
        )

    def get_ordered_classifications(
        self,
    ) -> list[ClassificationOfDriver]:
        """
        Get an ordered list of driver classifications.

        Returns
        -------
        list[ClassificationOfDriver]
            A list of ClassificationOfDriver objects ordered by their numeric keys
            in the classification_groups dictionary.

        Notes
        -----
        This method inverts the classification_groups dictionary using flip_dict
        and then retrieves the classifications in sequential order from 0 to the
        length of classification_groups.
        """
        flipped_class = flip_dict(dictionary=self.classification_groups)
        return [flipped_class[k] for k in range(len(self.classification_groups))]

    def get_ordered_drivers(
        self, classification: ClassificationOfDriver
    ) -> list[DriverName]:
        """
        Get an ordered list of driver names for a specific classification.

        Parameters
        ----------
        classification : ClassificationOfDriver
            The classification of drivers to retrieve and order.

        Returns
        -------
        list[DriverName]
            A sorted list of driver names, ordered by their corresponding values
            in the classification group mapping.

        Notes
        -----
        The method retrieves drivers from the classification group mapping and
        sorts them based on their associated values in ascending order.
        """
        class_drivers = self.maps[self.classification_groups[classification]]
        return sorted(class_drivers.keys(), key=lambda x: class_drivers[x])

    def get_all_ordered_drivers(self) -> list[DriverName]:
        """
        Get all drivers ordered by their index across all class maps.

        This method aggregates drivers from all class maps, combining their indices
        into a single ordering. When a driver appears in multiple class maps, the
        last occurrence's index is used.

        Returns
        -------
        list[DriverName]
            A list of driver names ordered by their combined index values, starting
            from index 0 to the total number of unique drivers.

        Notes
        -----
        The method works by:
        1. Collecting all driver-to-index mappings from all class maps
        2. Flipping the dictionary to create an index-to-driver mapping
        3. Returning drivers in sequential order by their indices
        """
        # Initialize an empty dictionary to hold the combined driver-to-index mapping
        full_map: dict[DriverName, int] = {}

        # Iterate through each classification's driver map
        for class_map in self.maps:
            # Merge each driver and its array index into the combined mapping
            # Note: If a driver appears in multiple classifications, the last
            # occurrence will overwrite previous index assignments
            for driver, idx in class_map.items():
                full_map[driver] = idx

        # Invert the mapping to get index-to-driver lookup (flip keys and values)
        flipped_full_map = flip_dict(dictionary=full_map)

        # Return drivers in sequential order by their indices (0, 1, 2, ...)
        return [flipped_full_map[idx] for idx in range(len(flipped_full_map))]

    def flip_dates(self) -> list[dict[int, datetime.date]]:
        """
        Flip the sequence of date dictionaries.

        Returns
        -------
        list[dict[int, datetime.date]]
            A list of dictionaries with flipped key-value pairs from the original dates.
            Each dictionary in the sequence has its integer keys and datetime.date values
            transposed using the flip_seq_dicts utility function.

        See Also
        --------
        flip_seq_dicts : Utility function that performs the flipping operation.
        """
        return flip_seq_dicts(seq_dicts=self.dates)

    def _lag_driver_dates(
        self, flipped_dates: dict[int, datetime.date], driver_lag: int, max_lag: int
    ) -> dict[datetime.date, int]:
        """
        Create a mapping of lagged dates to column indices for a driver variable.

        This method adjusts date indices based on the driver's specific lag relative to
        a maximum lag value, accounting for the temporal offset between different driver
        variables in the analysis.

        Parameters
        ----------
        flipped_dates : dict[int, datetime.date]
            A dictionary mapping column indices to their corresponding dates.
        driver_lag : int
            The specific lag (in time periods) associated with this driver variable.
        max_lag : int
            The maximum lag value across all driver variables in the system.

        Returns
        -------
        dict[datetime.date, int]
            A dictionary mapping adjusted dates to their corresponding column indices,
            accounting for the driver-specific lag offset.

        Notes
        -----
        The method iterates through valid column indices (excluding the last `max_lag`
        columns) and creates new date-to-index mappings by offsetting each date by
        the difference between `max_lag` and `driver_lag`.
        """
        # Initialize a new dictionary to store the date-to-index mapping after lag adjustment
        new_dates: dict[datetime.date, int] = {}

        # Iterate through valid column indices after accounting for max_lag truncation
        # The range is limited to (shape[1] - max_lag) to ensure we don't access out-of-bounds dates
        for k in range(self.arr.shape[1] - max_lag):
            # Map each output column index k to its corresponding date after lag adjustment
            # The date is fetched from the original flipped_dates by offsetting the index:
            # - Add max_lag to account for the overall truncation of the array
            # - Subtract driver_lag to apply this driver's specific lag offset
            # This effectively shifts the date by (max_lag - driver_lag) periods
            new_dates[flipped_dates[k + max_lag - driver_lag]] = k

        return new_dates

    def _lag_dates(
        self,
        best_lags: dict[ClassificationOfDriver, dict[DriverName, int]],
        max_lag: int,
    ) -> list[dict[datetime.date, int]]:
        """
        Lag dates for all drivers based on their optimal lag values.

        This method takes the best lag values for each driver and applies them to
        create a new set of lagged dates. It flips the dates, applies the appropriate
        lag for each driver, and returns a list of lagged date dictionaries.

        Parameters
        ----------
        best_lags : dict[ClassificationOfDriver, dict[DriverName, int]]
            A nested dictionary mapping driver classifications to driver names and
            their corresponding optimal lag values.
        max_lag : int
            The maximum lag value allowed for any driver.

        Returns
        -------
        list[dict[datetime.date, int]]
            A list of dictionaries where each dictionary maps dates to integer values
            for each driver after applying the appropriate lag. The order corresponds
            to the order returned by get_all_ordered_drivers().

        Notes
        -----
        This method first flattens the nested best_lags structure into a single
        dictionary mapping driver names to lags, then applies these lags to each
        driver's dates using _lag_driver_dates().
        """
        # Flatten the nested best_lags dictionary structure into a single dictionary
        # This combines all classifications into one driver-to-lag mapping for easier access
        all_lags: dict[DriverName, int] = {}
        for classification in best_lags.keys():
            # For each classification, extract all driver-lag pairs and add to the flat dictionary
            for driver, lag in best_lags[classification].items():
                all_lags[driver] = lag

        # Flip the dates dictionary structure from {date: idx} to {idx: date} for each driver
        # This makes it easier to look up dates by column index when applying lags
        flipped_dates = self.flip_dates()

        # Initialize list to store the new date mappings for each driver after lag adjustment
        new_dates: list[dict[datetime.date, int]] = []

        # Iterate through each driver in the order they appear across all classifications
        for idx, driver in enumerate(self.get_all_ordered_drivers()):
            # Apply lag transformation to this driver's dates
            # - Use the driver's position (idx) to get its flipped dates from the list
            # - Apply the driver's specific lag value from the flattened all_lags dict
            # - Use max_lag as reference point for alignment across all drivers
            new_dates.append(
                self._lag_driver_dates(
                    flipped_dates=flipped_dates[idx],
                    driver_lag=all_lags[driver],
                    max_lag=max_lag,
                )
            )

        return new_dates

    def apply_lag(
        self,
        best_lags: dict[ClassificationOfDriver, dict[DriverName, int]],
        max_lag: int,
    ) -> ClassifiedDriverGroups[ClassificationOfDriver]:
        """
        Apply lag transformations to driver data based on specified lag values.

        This method creates a new ClassifiedDriverGroups instance with driver data shifted
        according to the best lags for each classification and driver. The resulting array
        is truncated to account for the maximum lag applied.

        Parameters
        ----------
        best_lags : dict[ClassificationOfDriver, dict[DriverName, int]]
            A nested dictionary mapping classifications to dictionaries of driver names
            and their corresponding optimal lag values (in time steps). Lag values indicate
            how many time steps to shift each driver backward.
        max_lag : int
            The maximum lag value across all drivers and classifications. This determines
            how much the time dimension will be reduced in the output array.

        Returns
        -------
        ClassifiedDriverGroups[ClassificationOfDriver]
            A new ClassifiedDriverGroups instance with lagged driver data. The time
            dimension is reduced by max_lag to ensure all lagged series have the same
            length.

        Raises
        ------
        AssertionError
            If the classifications in best_lags don't match those in the
            ClassifiedDriverGroups instance, or if the driver names within each
            classification don't match.

        Notes
        -----
        - A lag of 0 means no shift is applied to that driver.
        - Positive lag values shift the driver data backward in time.
        - The output array has shape (n_drivers, n_timesteps - max_lag).
        - Dates are automatically adjusted to reflect the lag transformations.
        """
        # Validate that best_lags contains exactly the same classifications as this instance
        assert set(self.classification_groups.keys()) == set(best_lags.keys()), (
            'Given best lags does not contain the same classifications as ClassifiedDriverGroups.'
        )

        # Validate that each classification in best_lags contains the same drivers as in this instance
        for classification in self.classification_groups.keys():
            assert set(
                self.maps[self.classification_groups[classification]].keys()
            ) == set(best_lags[classification].keys()), (
                f'Given best lags does not conatin the same drivers as Classification '
                f'{classification} in ClassifiedDriverGroups.'
            )

        # Initialize a new array with truncated length (original_length - max_lag)
        # This ensures all lagged drivers align to the same temporal window
        new_drivers_arr = np.zeros(
            (self.arr.shape[0], self.arr.shape[1] - max_lag), dtype=self.np_dtype
        )

        # Apply the appropriate lag transformation to each driver in each classification
        for classification in self.classification_groups.keys():
            # Get the driver map for this classification
            class_map = self.maps[self.classification_groups[classification]]

            # Iterate through each driver in this classification
            for driver in class_map.keys():
                # Case 1: No lag (lag=0) - take data starting from max_lag position onwards
                # This aligns with the most-lagged driver's earliest valid data point
                if best_lags[classification][driver] == 0:
                    new_drivers_arr[class_map[driver]] = self.arr[
                        class_map[driver], max_lag:
                    ]
                # Case 2: Positive lag - shift the driver backwards in time
                # Start from (max_lag - driver_lag) and end at (-driver_lag) to exclude the last lag periods
                # This effectively aligns the lagged driver with the current time period
                else:
                    new_drivers_arr[class_map[driver]] = self.arr[
                        class_map[driver],
                        max_lag - best_lags[classification][driver] : -best_lags[
                            classification
                        ][driver],
                    ]

        # Return a new ClassifiedDriverGroups with lagged data, adjusted dates, and preserved metadata
        return ClassifiedDriverGroups(
            arr=new_drivers_arr,
            classification_groups=self.classification_groups,
            maps=self.maps,
            dates=self._lag_dates(best_lags=best_lags, max_lag=max_lag),
            np_dtype=self.np_dtype,
        )

    def apply_daterange(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> ClassifiedDriverGroups[ClassificationOfDriver]:
        """
        Apply a date range filter to the driver groups, extracting a subset of data.

        This method filters the driver data to only include dates within the specified
        range [start_date, end_date] (inclusive). It creates a new ClassifiedDriverGroups
        instance with the filtered data while preserving the classification structure.

        Parameters
        ----------
        start_date : datetime.date
            The start date of the range to extract (inclusive).
        end_date : datetime.date
            The end date of the range to extract (inclusive).

        Returns
        -------
        ClassifiedDriverGroups[ClassificationOfDriver]
            A new ClassifiedDriverGroups instance containing only the data within the
            specified date range, with updated array dimensions and date mappings.

        Raises
        ------
        AssertionError
            If either start_date or end_date is not available in the dates for any driver.

        Notes
        -----
        - The method validates that both start_date and end_date exist in the available
          dates for all drivers before proceeding with the extraction.
        - The resulting array will have dimensions (n_drivers, n_months) where n_months
          is the number of months between start_date and end_date (inclusive).
        - Date indices in the new instance are adjusted to start from 0.
        """
        # Validate that start_date and end_date exist in the available dates for all drivers
        # This ensures we can safely extract data for the requested range
        for classification in self.classification_groups.keys():
            class_map = self.maps[self.classification_groups[classification]]
            for driver in class_map.keys():
                # Check if start_date is present in this driver's date mapping
                assert start_date in self.dates[class_map[driver]].keys(), (
                    f'Start date for driver {driver} is not in available dates.'
                )
                # Check if end_date is present in this driver's date mapping
                assert end_date in self.dates[class_map[driver]].keys(), (
                    f'End date for driver {driver} is not in available dates.'
                )

        # Initialize a new array to hold the filtered data for all drivers
        # The time dimension is calculated as the number of months between start and end (inclusive)
        new_arr: ArrayF = np.zeros(
            (self.arr.shape[0], month_dif(start_date, end_date) + 1),
            dtype=self.np_dtype,
        )

        # Initialize a list to store the new date mappings for each driver
        new_dates: list[dict[datetime.date, int]] = []

        # Iterate through classifications and drivers in their original order
        # This ensures the output maintains the same driver ordering as the input
        for classification in self.get_ordered_classifications():
            class_map = self.maps[self.classification_groups[classification]]
            for driver in self.get_ordered_drivers(classification):
                # Get the array index corresponding to the start date for this driver
                start_idx = self.dates[class_map[driver]][start_date]
                # Get the array index corresponding to the end date, +1 to make the range inclusive
                end_idx = (
                    self.dates[class_map[driver]][end_date] + 1
                )  # +1 to include end_date

                # Copy the filtered slice of data from the original array into the new array
                new_arr[class_map[driver]] = self.arr[
                    class_map[driver], start_idx:end_idx
                ]

                # Create a new date-to-index mapping for this driver
                # Remap indices relative to the new start position (subtract start_idx)
                # Only include dates that fall within the filtered range [start_idx, end_idx)
                new_dates.append(
                    {
                        date: idx - start_idx
                        for date, idx in self.dates[class_map[driver]].items()
                        if start_idx <= idx < end_idx
                    }
                )

        # Return a new ClassifiedDriverGroups with the filtered data and updated date mappings
        return ClassifiedDriverGroups(
            arr=new_arr,
            classification_groups=self.classification_groups,
            maps=self.maps,
            dates=new_dates,
            np_dtype=self.np_dtype,
        )

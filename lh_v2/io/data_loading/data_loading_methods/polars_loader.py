import datetime
import pathlib as pth

import numpy as np
import polars as pl

import lh_v2.datatypes as dts
import lh_v2.params as params
from lh_v2.shared import BASE_NP_DTYPE, ArrayF
from lh_v2.util import ymd2pydate

from .abstract_class import AbstractDataLoader


class PolarsDataLoader(AbstractDataLoader):
    """
    Data loader implementation using Polars for efficient data loading and processing.

    This class provides methods to load account and driver data from Polars DataFrames
    or file paths (CSV/Parquet). It processes the data according to the specified
    lighthouse parameters and returns structured data types for use in forecasting.

    Parameters
    ----------
    lh_params : params.LighthouseParams
        Lighthouse parameters containing configuration for data loading, including
        segment, region, and account specifications.

    Methods
    -------
    load_account_data(source)
        Load and process account data from a Polars DataFrame or file path.
    load_driver_data(source)
        Load and process driver data from a Polars DataFrame or file path.

    See Also
    --------
    AbstractDataLoader : Parent abstract class defining the data loader interface.

    Examples
    --------
    >>> loader = PolarsDataLoader(lh_params)
    >>> accounts = loader.load_account_data('data/accounts.parquet')
    >>> drivers = loader.load_driver_data('data/drivers.csv')
    """

    def __init__(
        self,
        lh_params: params.LighthouseParams,
        np_dtype: type[np.floating] = BASE_NP_DTYPE,
    ) -> None:
        super().__init__(lh_params)
        self.np_dtype = np_dtype
        return

    def load_account_data(
        self, source: pl.DataFrame | pth.Path
    ) -> dts.AccountGroupInfo:
        """
        Load and process account data from a Polars DataFrame or file.

        This method loads account data, filters it according to the specified segment,
        region, and account types, and structures it into an AccountGroupInfo object.
        Data is aggregated by period and organized into arrays with corresponding
        date mappings.

        Parameters
        ----------
        source : pl.DataFrame | pth.Path
            Either a Polars DataFrame containing account data or a path to a CSV/Parquet
            file. If a path, the file will be loaded based on its extension.

        Returns
        -------
        dts.AccountGroupInfo
            Structured account data containing arrays, account mappings, dates,
            and hierarchy information (segment and region).

        Raises
        ------
        FileNotFoundError
            If the source path does not exist.
        AssertionError
            If required columns are missing from the DataFrame or if the source
            is not a valid Polars DataFrame.

        Notes
        -----
        Required columns in the source data:
        - account_key: Account identifier
        - period_key: Period/date identifier
        - bu_key: Business unit key
        - plant_key: Plant/location key
        - value: Account value

        The method filters data based on segment, region, and account specifications
        from lh_params, then aggregates values by period.

        Examples
        --------
        >>> loader = PolarsDataLoader(lh_params)
        >>> accounts = loader.load_account_data('data/accounts.parquet')
        >>> accounts.arr.shape
        (5, 36)  # 5 accounts, 36 time periods
        """
        # Load data from file if path is provided
        if isinstance(source, pth.Path):
            # Verify file exists before attempting to load
            if not source.exists():
                raise FileNotFoundError(f'Account data file not found: {source}')
            # Load CSV files using Polars CSV reader
            if source.suffix.lower() == '.csv':
                source = pl.read_csv(source)
            # Load Parquet files using Polars Parquet reader
            elif source.suffix.lower() == '.parquet':
                source = pl.read_parquet(source)

        # Validate that source is now a Polars DataFrame
        assert isinstance(source, pl.DataFrame), 'Source must be a Polars DataFrame.'
        # Verify all required columns are present in the DataFrame
        assert 'account_key' in source.columns, (
            "DataFrame must contain 'account_key' column."
        )
        assert 'period_key' in source.columns, (
            "DataFrame must contain 'period_key' column."
        )
        assert 'bu_key' in source.columns, "DataFrame must contain 'bu_key' column."
        assert 'plant_key' in source.columns, (
            "DataFrame must contain 'plant_key' column."
        )
        assert 'value' in source.columns, "DataFrame must contain 'value' column."

        polars_filters: list[pl.Expr] = []

        # Determine which column contains the segment/product information
        seg_key = _get_product_key(df=source, product_type=self.lh_params.segment)

        if seg_key is not None:
            polars_filters.append(pl.col(seg_key) == self.lh_params.segment)

        # Determine which column contains the region/location information
        reg_key = _get_location_key(df=source, location_type=self.lh_params.region)

        if reg_key is not None:
            polars_filters.append(pl.col(reg_key) == self.lh_params.region)

        polars_filters.append(pl.col('account_key').is_in(self.lh_params.accounts))

        # Filter data to specified segment, region, and accounts, then aggregate by period
        filtered_df = (
            source.filter(polars_filters)
            .group_by(
                ['period_key', seg_key, reg_key, 'account_key']
            )  # Group by key dimensions
            .agg(pl.col('value').sum())  # Aggregate values by summing
        )

        # Initialize array to store account data: rows = accounts, columns = time periods
        arr_accounts: ArrayF = np.zeros(
            (len(self.lh_params.accounts), filtered_df.unique('period_key').shape[0]),
            dtype=self.np_dtype,
        )

        # Build account mapping and populate array with account values
        acc_map: dict[dts.AccountType, int] = {}
        for i, account in enumerate(self.lh_params.accounts):
            # Filter to specific account and sort by period for chronological order
            account_df = filtered_df.filter(pl.col('account_key') == account).sort(
                'period_key'
            )
            # Copy account values into the array for this account (row i)
            arr_accounts[i, :] = account_df['value'].to_numpy().astype(self.np_dtype)
            # Map account type to its row index in the array
            acc_map[dts.AccountType(account)] = i

        # Extract unique period keys and create date-to-index mapping
        dates_vals = list(
            filtered_df.select('period_key').unique().sort('period_key')['period_key']
        )
        dates_dict: dict[datetime.date, int] = {}
        for i, date in enumerate(dates_vals):
            # Convert period key to Python date and map to column index
            dates_dict[ymd2pydate(date)] = i

        # Create a copy of the date mapping for each account (all accounts share same dates)
        dates_info: list[dict[datetime.date, int]] = []
        for _ in range(len(self.lh_params.accounts)):
            dates_info.append(dates_dict.copy())

        # Return structured AccountGroupInfo with all processed data
        return dts.AccountGroupInfo(
            arr=arr_accounts,
            account_map=acc_map,
            dates=dates_info,
            segment_type=dts.HierarchyTree(
                name=dts.ProductType(self.lh_params.segment)
            ),
            region_type=dts.HierarchyTree(name=dts.LocationType(self.lh_params.region)),
            np_dtype=self.np_dtype,
        )

    def load_driver_data(
        self, source: pl.DataFrame | pth.Path
    ) -> dts.ClassifiedDriverGroups[dts.DriverClassification]:
        """
        Load and process driver data from a Polars DataFrame or file.

        This method loads driver data, organizes it by classification, and structures
        it into a ClassifiedDriverGroups object. Each driver classification is
        processed separately, with drivers grouped and organized with their time
        series data and date mappings.

        Parameters
        ----------
        source : pl.DataFrame | pth.Path
            Either a Polars DataFrame containing driver data or a path to a CSV/Parquet
            file. If a path, the file will be loaded based on its extension.

        Returns
        -------
        dts.ClassifiedDriverGroups[dts.DriverClassification]
            Structured driver data organized by classification, containing arrays,
            driver mappings, and dates for each classification group.

        Raises
        ------
        FileNotFoundError
            If the source path does not exist.
        AssertionError
            If required columns are missing from the DataFrame, if the source
            is not a valid Polars DataFrame, or if no driver classifications are found.

        Notes
        -----
        Required columns in the source data:
        - driver_unique_name: Unique identifier for each driver
        - driver_classification: Classification category for the driver
        - date: Date for the driver value
        - driver_value: The driver's value

        Optional columns:
        - denorm: Denormalization factor (if present, multiplied with driver_value)

        The method processes each driver classification separately, creating a
        DriverGroup for each classification and combining them into a
        ClassifiedDriverGroups structure.

        Examples
        --------
        >>> loader = PolarsDataLoader(lh_params)
        >>> drivers = loader.load_driver_data('data/drivers.csv')
        >>> primary_drivers = drivers[DriverClassification('PRIMARY')]
        >>> primary_drivers.arr.shape
        (10, 36)  # 10 drivers, 36 time periods
        """
        # Load data from file if path is provided
        if isinstance(source, pth.Path):
            # Verify file exists before attempting to load
            if not source.exists():
                raise FileNotFoundError(f'Account data file not found: {source}')
            # Load CSV files using Polars CSV reader
            if source.suffix.lower() == '.csv':
                source = pl.read_csv(source)
            # Load Parquet files using Polars Parquet reader
            elif source.suffix.lower() == '.parquet':
                source = pl.read_parquet(source)

        # Validate that source is now a Polars DataFrame
        assert isinstance(source, pl.DataFrame), 'Source must be a Polars DataFrame.'
        # Verify all required columns are present in the DataFrame
        assert 'driver_unique_name' in source.columns, (
            "DataFrame must contain 'driver_unique_name' column."
        )
        assert 'driver_classification' in source.columns, (
            "DataFrame must contain 'driver_classification' column."
        )
        assert 'date' in source.columns, "DataFrame must contain 'date' column."

        driver_value_key = _get_driver_value_key(df=source)

        # Apply denormalization if denorm column is present
        if 'denorm' in source.columns:
            # Multiply driver_value by denorm factor to get actual values
            source = source.with_columns(
                (pl.col(driver_value_key) * pl.col('denorm')).alias(driver_value_key)
            )

        # Extract unique driver classifications from the data
        driver_classes = source['driver_classification'].unique().sort().to_list()
        assert len(driver_classes) > 0, 'Driver classes list cannot be empty.'

        # Build global date-to-index mapping for all drivers
        dates_vals = source['date'].unique().sort().to_list()
        dates_dict: dict[datetime.date, int] = {}
        for i, date in enumerate(dates_vals):
            # Convert date to Python date object and map to column index
            dates_dict[ymd2pydate(date)] = i

        # Initialize containers for classified driver groups
        classified_driver_group_lst: list[dts.DriverGroup] = []
        classifications: dict[dts.DriverClassification, int] = {}

        # Process each driver classification separately
        for idx, driver_class in enumerate(driver_classes):
            # Map classification to its index in the list
            classifications[dts.DriverClassification(driver_class)] = idx

            # Filter data to current classification
            class_df = source.filter(pl.col('driver_classification') == driver_class)
            # Extract unique driver names within this classification
            driver_names = class_df['driver_unique_name'].unique().sort().to_list()

            # Initialize array for this classification: rows = drivers, columns = dates
            class_arr: ArrayF = np.zeros(
                (len(driver_names), len(dates_vals)), dtype=self.np_dtype
            )
            # Initialize mapping and date structures for this classification
            class_driver_map: dict[dts.DriverName, int] = {}
            class_dates: list[dict[datetime.date, int]] = []

            # Process each driver in this classification
            for idx2, driver_name in enumerate(driver_names):
                # Map driver name to its row index in the classification array
                class_driver_map[dts.DriverName(driver_name)] = idx2

                # Filter to specific driver and sort chronologically
                driver_df = class_df.filter(
                    pl.col('driver_unique_name') == driver_name
                ).sort('date')
                # Add date mapping for this driver (copy of global mapping)
                class_dates.append(dates_dict.copy())
                # Copy driver values into the array for this driver (row idx2)
                class_arr[idx2, :] = (
                    driver_df[driver_value_key].to_numpy().astype(self.np_dtype)
                )

            # Create DriverGroup for this classification with all its drivers
            classified_driver_group_lst.append(
                dts.DriverGroup(
                    arr=class_arr,
                    map=class_driver_map,
                    dates=class_dates,
                    np_dtype=self.np_dtype,
                )
            )

        # Combine all classification groups into a single ClassifiedDriverGroups structure
        return dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ].from_driver_group_lst(
            driver_group_lst=classified_driver_group_lst,
            classification_groups=classifications,
        )


def _get_product_key(
    df: pl.DataFrame,
    product_type: dts.ProductType,
) -> str | None:
    """
    Get the product key column name based on the product type.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the data.
    product_type : dts.ProductType
        The type of product (e.g., PRODUCT_GROUP, PRODUCT_KEY).

    Returns
    -------
    str
        The column name corresponding to the product type.

    Raises
    ------
    ValueError
        If the product type is not recognized or the corresponding column is not found.
    """
    # If product type is 'total' or 'all', no filtering is needed
    if product_type.lower() in [dts.ProductType('total'), dts.ProductType('all')]:
        return None

    # Check if product type is in the business unit column
    if product_type in df['bu_key'].unique().to_list():
        return 'bu_key'

    # If not in bu_key, check product_group column (must exist)
    assert 'product_group' in df.columns, (
        "DataFrame must contain 'product_group' column if given product type is not in 'bu_key'."
    )
    if product_type in df['product_group'].unique().to_list():
        return 'product_group'

    # If not in bu_key or product_group, check product_key column (must exist)
    assert 'product_key' in df.columns, (
        "DataFrame must contain 'product_key' column if "
        "given product type is not in 'bu_key' or 'product_group'."
    )
    if product_type in df['product_key'].unique().to_list():
        return 'product_key'

    # Product type not found in any expected column
    raise ValueError(
        f"Product type '{product_type}' not found in 'bu_key', "
        "'product_group', or 'product_key' columns."
    )


def _get_location_key(
    df: pl.DataFrame,
    location_type: dts.LocationType,
) -> str | None:
    """
    Get the region key column name based on the region type.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the data.
    region_type : dts.RegionType
        The type of region (e.g., REGION_GROUP, REGION_KEY).

    Returns
    -------
    str
        The column name corresponding to the region type.

    Raises
    ------
    ValueError
        If the region type is not recognized or the corresponding column is not found.
    """
    # If region type is 'total' or 'all', no filtering is needed
    if location_type.lower() in [dts.LocationType('total'), dts.LocationType('all')]:
        return None

    # Check if location type is in the plant_key column
    if location_type in df['plant_key'].unique().to_list():
        return 'plant_key'

    # Location type not found in plant_key column
    raise ValueError(
        f"Location type '{location_type}' not found in 'plant_key' column."
    )


def _get_driver_value_key(
    df: pl.DataFrame,
) -> str:
    """
    Get the driver value column name.

    Parameters
    ----------
    df : pl.DataFrame
        The Polars DataFrame containing the data.

    Returns
    -------
    str
        The column name corresponding to the driver value.

    Raises
    ------
    ValueError
        If the driver value column is not found.
    """
    # Check if 'driver_value' column exists
    if 'driver_value' in df.columns:
        return 'driver_value'

    # Check if 'value' column exists as an alternative
    if 'value' in df.columns:
        return 'value'

    # Driver value column not found
    raise ValueError("DataFrame must contain 'driver_value' or 'value' column.")

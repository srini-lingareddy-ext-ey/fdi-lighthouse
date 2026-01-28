import pathlib as pth

import numpy as np

from lh_v2.datatypes import (
    AccountGroupInfo,
    AccountType,
    ClassifiedDriverGroups,
    DriverClassification,
    LocationType,
    ProductType,
)
from lh_v2.io.data_loading.data_loading_methods import PolarsDataLoader
from lh_v2.params import LighthouseParams
from lh_v2.shared import BASE_NP_DTYPE


def read_test_acc_data(
    file_path: pth.Path,
    accounts: list[AccountType] = [
        AccountType('volume'),
        AccountType('net_revenue'),
        AccountType('cogs_total'),
        AccountType('gross_margin'),
    ],
    product: ProductType = ProductType('Residential'),
    location: LocationType = LocationType('North America'),
    np_dtype: type[np.floating] = BASE_NP_DTYPE,
) -> AccountGroupInfo:
    """Read test account group data from a Parquet file."""
    lh_params = LighthouseParams()
    lh_params.accounts = accounts
    lh_params.segment = product
    lh_params.region = location

    data_loader = PolarsDataLoader(lh_params=lh_params, np_dtype=np_dtype)
    return data_loader.load_account_data(source=file_path)


def read_test_driver_data(
    file_path: pth.Path,
    np_dtype: type[np.floating] = BASE_NP_DTYPE,
) -> ClassifiedDriverGroups[DriverClassification]:
    """Read test driver data from a Parquet file."""
    lh_params = LighthouseParams()
    data_loader = PolarsDataLoader(lh_params=lh_params, np_dtype=np_dtype)
    return data_loader.load_driver_data(source=file_path)

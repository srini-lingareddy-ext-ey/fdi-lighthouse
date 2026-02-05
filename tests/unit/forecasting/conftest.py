import datetime
from typing import Any

import numpy as np
import pytest
from dateutil import relativedelta

from lh_v2.datatypes import (
    AccountGroupInfo,
    AccountInfo,
    AccountType,
    Driver,
    DriverGroup,
    DriverName,
    HierarchyTree,
    LocationType,
    ProductType,
)
from lh_v2.params import LighthouseParams, parse_yaml
from lh_v2.shared import BASE_NP_DTYPE, ArrayF


@pytest.fixture
def example_dates() -> list[datetime.date]:
    """Return example dates for testing."""
    start_date = datetime.date(2020, 1, 1)
    return [
        start_date + relativedelta.relativedelta(months=i) for i in range(36)
    ]  # Monthly dates for three years


@pytest.fixture
def example_forecasting_params() -> LighthouseParams:
    """Return example LighthouseParams for testing."""
    yml_dict: dict[str, Any] = {
        'use_default_params': True,
        'accounts': [
            'account_1',
            'account_2',
            'account_3',
        ],
        'segment': 'product_A',
        'region': 'location_X',
        'general_params': {
            'training_start_date': '2020-01-01',
            'training_end_date': '2022-12-01',
            'validation_start_date': '2023-01-01',
            'validation_end_date': '2023-06-01',
            'testing_start_date': '2023-07-01',
            'testing_end_date': '2023-12-01',
        },
    }
    return parse_yaml(yml_dict)


@pytest.fixture
def driv_arr_1(example_dates: list[datetime.date]) -> ArrayF:
    """Return example driver array 1 for testing."""
    n_dates = len(example_dates)
    arr: ArrayF = np.linspace(10, 45, n_dates, dtype=BASE_NP_DTYPE)
    np.random.seed(0)
    noise = np.random.normal(0, 2, n_dates).astype(BASE_NP_DTYPE)
    return arr + noise


@pytest.fixture
def driv_arr_2(example_dates: list[datetime.date]) -> ArrayF:
    """Return example driver array 2 for testing."""
    n_dates = len(example_dates)
    arr: ArrayF = np.linspace(20, 60, n_dates, dtype=BASE_NP_DTYPE)
    np.random.seed(1)
    noise = np.random.normal(0, 3, n_dates).astype(BASE_NP_DTYPE)
    return arr + noise


@pytest.fixture
def driv_arr_3(example_dates: list[datetime.date]) -> ArrayF:
    """Return example driver array 3 for testing."""
    n_dates = len(example_dates)
    arr: ArrayF = np.linspace(15, 50, n_dates, dtype=BASE_NP_DTYPE)
    np.random.seed(2)
    noise = np.random.normal(0, 2.5, n_dates).astype(BASE_NP_DTYPE)
    return arr + noise


@pytest.fixture
def driv_arr_4(example_dates: list[datetime.date]) -> ArrayF:
    """Return example driver array 4 for testing."""
    n_dates = len(example_dates)
    arr: ArrayF = np.linspace(5, 30, n_dates, dtype=BASE_NP_DTYPE)
    np.random.seed(3)
    noise = np.random.normal(0, 1.5, n_dates).astype(BASE_NP_DTYPE)
    return arr + noise


@pytest.fixture
def driver1_info(driv_arr_1: ArrayF, example_dates: list[datetime.date]) -> Driver:
    """Return an example Driver for testing."""
    driver_name = DriverName('driver_1')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    return Driver(
        name=driver_name,
        arr=driv_arr_1,
        dates=dates_dict,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def driver2_info(driv_arr_2: ArrayF, example_dates: list[datetime.date]) -> Driver:
    """Return an example Driver for testing."""
    driver_name = DriverName('driver_2')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    return Driver(
        name=driver_name,
        arr=driv_arr_2,
        dates=dates_dict,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def driver3_info(driv_arr_3: ArrayF, example_dates: list[datetime.date]) -> Driver:
    """Return an example Driver for testing."""
    driver_name = DriverName('driver_3')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    return Driver(
        name=driver_name,
        arr=driv_arr_3,
        dates=dates_dict,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def driver4_info(driv_arr_4: ArrayF, example_dates: list[datetime.date]) -> Driver:
    """Return an example Driver for testing."""
    driver_name = DriverName('driver_4')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    return Driver(
        name=driver_name,
        arr=driv_arr_4,
        dates=dates_dict,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def driver_group1(
    example_dates: list[datetime.date],
    driv_arr_1: ArrayF,
    driv_arr_2: ArrayF,
    driv_arr_3: ArrayF,
) -> DriverGroup:
    """Return an example DriverGroup for testing."""
    drivers = [
        DriverName('driver_1'),
        DriverName('driver_2'),
        DriverName('driver_3'),
    ]

    arr_drivers: ArrayF = np.vstack(
        [driv_arr_1, driv_arr_2, driv_arr_3],
        dtype=BASE_NP_DTYPE,
    )

    drivers_map: dict[DriverName, int] = {
        drivers[0]: 0,
        drivers[1]: 1,
        drivers[2]: 2,
    }

    dates_lst: list[dict[datetime.date, int]] = []
    for _ in range(arr_drivers.shape[0]):
        dates_dict: dict[datetime.date, int] = {}
        for idx, date in enumerate(example_dates):
            dates_dict[date] = idx
        dates_lst.append(dates_dict)

    return DriverGroup(
        arr=arr_drivers,
        map=drivers_map,
        dates=dates_lst,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def acc_arr_1(example_dates: list[datetime.date]) -> ArrayF:
    """Return example account array 1 for testing."""
    n_dates = len(example_dates)
    arr: ArrayF = np.linspace(100, 300, n_dates, dtype=BASE_NP_DTYPE)
    np.random.seed(4)
    noise = np.random.normal(0, 10, n_dates).astype(BASE_NP_DTYPE)
    return arr + noise


@pytest.fixture
def acc_arr_2(example_dates: list[datetime.date]) -> ArrayF:
    """Return example account array 2 for testing."""
    n_dates = len(example_dates)
    arr: ArrayF = np.linspace(150, 350, n_dates, dtype=BASE_NP_DTYPE)
    np.random.seed(5)
    noise = np.random.normal(0, 12, n_dates).astype(BASE_NP_DTYPE)
    return arr + noise


@pytest.fixture
def acc_arr_3(example_dates: list[datetime.date]) -> ArrayF:
    """Return example account array 3 for testing."""
    n_dates = len(example_dates)
    arr: ArrayF = np.linspace(120, 320, n_dates, dtype=BASE_NP_DTYPE)
    np.random.seed(6)
    noise = np.random.normal(0, 11, n_dates).astype(BASE_NP_DTYPE)
    return arr + noise


@pytest.fixture
def account1_info(acc_arr_1: ArrayF, example_dates: list[datetime.date]) -> AccountInfo:
    """Return an example AccountInfo for testing."""
    account_name = AccountType('account_1')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    product_type = ProductType('product_A')
    location_type = LocationType('location_X')

    return AccountInfo(
        arr=acc_arr_1,
        dates=dates_dict,
        account_type=account_name,
        segment_type=HierarchyTree(product_type),
        region_type=HierarchyTree(location_type),
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def account2_info(acc_arr_2: ArrayF, example_dates: list[datetime.date]) -> AccountInfo:
    """Return an example AccountInfo for testing."""
    account_name = AccountType('account_2')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    product_type = ProductType('product_A')
    location_type = LocationType('location_X')

    return AccountInfo(
        arr=acc_arr_2,
        dates=dates_dict,
        account_type=account_name,
        segment_type=HierarchyTree(product_type),
        region_type=HierarchyTree(location_type),
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def account3_info(acc_arr_3: ArrayF, example_dates: list[datetime.date]) -> AccountInfo:
    """Return an example AccountInfo for testing."""
    account_name = AccountType('account_3')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    product_type = ProductType('product_A')
    location_type = LocationType('location_X')

    return AccountInfo(
        arr=acc_arr_3,
        dates=dates_dict,
        account_type=account_name,
        segment_type=HierarchyTree(product_type),
        region_type=HierarchyTree(location_type),
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def account_group_1(
    account1_info: AccountInfo,
    account2_info: AccountInfo,
    account3_info: AccountInfo,
) -> AccountGroupInfo:
    """Return an example AccountGroupInfo for testing."""
    return AccountGroupInfo.from_account_lst(
        [account1_info, account2_info, account3_info]
    )

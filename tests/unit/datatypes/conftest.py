import datetime

import numpy as np
import pytest

from lh_v2.datatypes import (
    AccountGroupInfo,
    AccountInfo,
    AccountType,
    ClassifiedDriverGroups,
    Driver,
    DriverClassification,
    DriverGroup,
    DriverName,
    HierarchyTree,
    LocationType,
    ProductType,
)
from lh_v2.shared import BASE_NP_DTYPE, ArrayF


@pytest.fixture
def example_dates() -> list[datetime.date]:
    """Return example dates for testing."""
    return [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 2, 1),
        datetime.date(2024, 3, 1),
        datetime.date(2024, 4, 1),
        datetime.date(2024, 5, 1),
        datetime.date(2024, 6, 1),
    ]


@pytest.fixture
def acc_arr1() -> ArrayF:
    """Return an example account array for testing."""
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def acc_arr2() -> ArrayF:
    """Return an example account array for testing."""
    return np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def acc_arr3() -> ArrayF:
    """Return an example account array for testing."""
    return np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def account_info1(acc_arr1: ArrayF, example_dates: list[datetime.date]) -> AccountInfo:
    """Return an example AccountInfo for testing."""
    account_type = AccountType('account_1')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    product_type = ProductType('product_A')
    location_type = LocationType('location_X')

    return AccountInfo(
        arr=acc_arr1,
        dates=dates_dict,
        account_type=account_type,
        segment_type=HierarchyTree(product_type),
        region_type=HierarchyTree(location_type),
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def account_info2(acc_arr2: ArrayF, example_dates: list[datetime.date]) -> AccountInfo:
    """Return an example AccountInfo for testing."""
    account_type = AccountType('account_2')

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    product_type = ProductType('product_A')
    location_type = LocationType('location_X')

    return AccountInfo(
        arr=acc_arr2,
        dates=dates_dict,
        account_type=account_type,
        segment_type=HierarchyTree(product_type),
        region_type=HierarchyTree(location_type),
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def account_group_info_example(
    acc_arr1: ArrayF,
    acc_arr2: ArrayF,
    acc_arr3: ArrayF,
    example_dates: list[datetime.date],
) -> AccountGroupInfo:
    """Return an example AccountGroupInfo for testing."""
    account_types = [
        AccountType('account_1'),
        AccountType('account_2'),
        AccountType('account_3'),
    ]

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    arr_accounts: ArrayF = np.vstack(
        [acc_arr1, acc_arr2, acc_arr3], dtype=BASE_NP_DTYPE
    )

    dates_lst: list[dict[datetime.date, int]] = []
    for _ in range(arr_accounts.shape[0]):
        dates_lst.append(dates_dict.copy())

    product_type = ProductType('product_A')
    location_type = LocationType('location_X')

    return AccountGroupInfo(
        arr=arr_accounts,
        account_map={account_types[i]: i for i in range(len(account_types))},
        dates=dates_lst,
        segment_type=HierarchyTree(product_type),
        region_type=HierarchyTree(location_type),
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def driv_arr_1() -> ArrayF:
    """Return an example driver array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def driv_arr_2() -> ArrayF:
    """Return an example driver array for testing."""
    return np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def driv_arr_3() -> ArrayF:
    """Return an example driver array for testing."""
    return np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def driv_arr_4() -> ArrayF:
    """Return an example driver array for testing."""
    return np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def driv_arr_5() -> ArrayF:
    """Return an example driver array for testing."""
    return np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=BASE_NP_DTYPE)


@pytest.fixture
def driver1(driv_arr_1: ArrayF, example_dates: list[datetime.date]) -> Driver:
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
def driver2(driv_arr_2: ArrayF, example_dates: list[datetime.date]) -> Driver:
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
def driver3(driv_arr_3: ArrayF, example_dates: list[datetime.date]) -> Driver:
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
def driver_group1(
    driv_arr_1: ArrayF,
    driv_arr_2: ArrayF,
    driv_arr_3: ArrayF,
    example_dates: list[datetime.date],
) -> DriverGroup:
    """Return an example DriverGroup for testing."""
    drivers = [
        DriverName('driver_1'),
        DriverName('driver_2'),
        DriverName('driver_3'),
    ]
    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    arr_drivers: ArrayF = np.vstack(
        [driv_arr_1, driv_arr_2, driv_arr_3], dtype=BASE_NP_DTYPE
    )

    drivers_map: dict[DriverName, int] = {
        drivers[0]: 0,
        drivers[1]: 1,
        drivers[2]: 2,
    }

    dates_lst: list[dict[datetime.date, int]] = []
    for _ in range(arr_drivers.shape[0]):
        dates_lst.append(dates_dict.copy())

    return DriverGroup(
        arr=arr_drivers,
        map=drivers_map,
        dates=dates_lst,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def driver_group2(
    driv_arr_4: ArrayF, driv_arr_5: ArrayF, example_dates: list[datetime.date]
) -> DriverGroup:
    """Return an example DriverGroup for testing."""
    drivers = [
        DriverName('driver_4'),
        DriverName('driver_5'),
    ]

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    arr_drivers: ArrayF = np.vstack([driv_arr_4, driv_arr_5], dtype=BASE_NP_DTYPE)

    drivers_map: dict[DriverName, int] = {
        drivers[0]: 0,
        drivers[1]: 1,
    }

    dates_lst: list[dict[datetime.date, int]] = []
    for _ in range(arr_drivers.shape[0]):
        dates_lst.append(dates_dict.copy())

    return DriverGroup(
        arr=arr_drivers,
        map=drivers_map,
        dates=dates_lst,
        np_dtype=BASE_NP_DTYPE,
    )


@pytest.fixture
def classified_driver_groups_example(
    driv_arr_1: ArrayF,
    driv_arr_2: ArrayF,
    driv_arr_3: ArrayF,
    driv_arr_4: ArrayF,
    driv_arr_5: ArrayF,
    example_dates: list[datetime.date],
) -> ClassifiedDriverGroups[DriverClassification]:
    """Return an example DriverGroup for testing."""
    classifications = [
        DriverClassification('class_1'),
        DriverClassification('class_2'),
    ]

    drivers = [
        DriverName('driver_1'),
        DriverName('driver_2'),
        DriverName('driver_3'),
        DriverName('driver_4'),
        DriverName('driver_5'),
    ]

    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(example_dates):
        dates_dict[date] = idx

    arr_drivers: ArrayF = np.vstack(
        [driv_arr_1, driv_arr_2, driv_arr_3, driv_arr_4, driv_arr_5],
        dtype=BASE_NP_DTYPE,
    )

    class_map: dict[DriverClassification, int] = {
        classifications[0]: 0,
        classifications[1]: 1,
    }

    driver_maps: list[dict[DriverName, int]] = [
        {
            drivers[0]: 0,
            drivers[1]: 1,
            drivers[2]: 2,
        },
        {
            drivers[3]: 3,
            drivers[4]: 4,
        },
    ]

    dates_lst: list[dict[datetime.date, int]] = []
    for _ in range(arr_drivers.shape[0]):
        dates_lst.append(dates_dict.copy())

    return ClassifiedDriverGroups[DriverClassification](
        arr=arr_drivers,
        classification_groups=class_map,
        maps=driver_maps,
        dates=dates_lst,
        np_dtype=BASE_NP_DTYPE,
    )

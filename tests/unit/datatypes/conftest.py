import datetime

import numpy as np
import pytest

from lh_v2.datatypes import (
    AccountGroupInfo,
    AccountType,
    ClassifiedDriverGroups,
    DriverClassification,
    DriverName,
    HierarchyTree,
    LocationType,
    ProductType,
)
from lh_v2.shared import BASE_NP_DTYPE, ArrayF


@pytest.fixture
def account_group_info_example() -> AccountGroupInfo:
    """Return an example AccountGroupInfo for testing."""
    account_types = [
        AccountType('account_1'),
        AccountType('account_2'),
        AccountType('account_3'),
    ]

    arr_1: ArrayF = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=BASE_NP_DTYPE)
    arr_2: ArrayF = np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0], dtype=BASE_NP_DTYPE)
    arr_3: ArrayF = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=BASE_NP_DTYPE)

    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 2),
        datetime.date(2024, 1, 3),
        datetime.date(2024, 1, 4),
        datetime.date(2024, 1, 5),
        datetime.date(2024, 1, 6),
    ]
    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(dates):
        dates_dict[date] = idx

    arr_accounts: ArrayF = np.vstack([arr_1, arr_2, arr_3], dtype=BASE_NP_DTYPE)

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
def classified_driver_groups_example() -> ClassifiedDriverGroups[DriverClassification]:
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

    arr_1: ArrayF = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=BASE_NP_DTYPE)
    arr_2: ArrayF = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=BASE_NP_DTYPE)
    arr_3: ArrayF = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=BASE_NP_DTYPE)
    arr_4: ArrayF = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=BASE_NP_DTYPE)
    arr_5: ArrayF = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=BASE_NP_DTYPE)

    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 2),
        datetime.date(2024, 1, 3),
        datetime.date(2024, 1, 4),
        datetime.date(2024, 1, 5),
        datetime.date(2024, 1, 6),
    ]
    dates_dict: dict[datetime.date, int] = {}
    for idx, date in enumerate(dates):
        dates_dict[date] = idx

    arr_drivers: ArrayF = np.vstack(
        [arr_1, arr_2, arr_3, arr_4, arr_5], dtype=BASE_NP_DTYPE
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

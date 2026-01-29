import datetime

import numpy as np
import pytest

import lh_v2.datatypes as dts
from lh_v2.shared import ArrayF


@pytest.mark.unit
@pytest.mark.datatypes
class TestAccountInfo:
    def test_account_info1(
        self,
        account_info1: dts.AccountInfo,
        acc_arr1: ArrayF,
        example_dates: list[datetime.date],
    ):
        # General asserts
        assert isinstance(account_info1, dts.AccountInfo)
        assert account_info1.arr.shape[0] == 6  # 6 time periods
        assert len(account_info1.dates) == 6

        # Account type asserts
        assert account_info1.account_type == dts.AccountType('account_1')

        # Segment type asserts
        assert account_info1.segment_type.name == dts.ProductType('product_A')

        # Region type asserts
        assert account_info1.region_type.name == dts.LocationType('location_X')

        # Check array values
        assert np.allclose(account_info1.arr, acc_arr1)

        # Check dates mapping
        expected_date_map = {date: idx for idx, date in enumerate(example_dates)}
        assert account_info1.dates == expected_date_map

    def test_account_info2(
        self,
        account_info2: dts.AccountInfo,
        acc_arr2: ArrayF,
        example_dates: list[datetime.date],
    ):
        # General asserts
        assert isinstance(account_info2, dts.AccountInfo)
        assert account_info2.arr.shape[0] == 6  # 6 time periods
        assert len(account_info2.dates) == 6

        # Account type asserts
        assert account_info2.account_type == dts.AccountType('account_2')

        # Segment type asserts
        assert account_info2.segment_type.name == dts.ProductType('product_A')

        # Region type asserts
        assert account_info2.region_type.name == dts.LocationType('location_X')

        # Check array values
        assert np.allclose(account_info2.arr, acc_arr2)

        # Check dates mapping
        expected_date_map = {date: idx for idx, date in enumerate(example_dates)}
        assert account_info2.dates == expected_date_map

    def test_account_info1_apply_lag(
        self,
        account_info1: dts.AccountInfo,
        acc_arr1: ArrayF,
        example_dates: list[datetime.date],
    ):
        # Apply lag of 2
        lagged_info = account_info1.apply_lag(max_lag=2)

        # General asserts
        assert isinstance(lagged_info, dts.AccountInfo)
        assert lagged_info.arr.shape[0] == 4  # 6 original - 2 lag
        assert len(lagged_info.dates) == 4

        # Check array values after lagging
        assert np.allclose(lagged_info.arr, acc_arr1[2:])

        # Check dates mapping after lagging
        expected_date_map = {date: idx for idx, date in enumerate(example_dates[2:])}
        assert lagged_info.dates == expected_date_map

    def test_account_info2_apply_lag(
        self,
        account_info2: dts.AccountInfo,
        acc_arr2: ArrayF,
        example_dates: list[datetime.date],
    ):
        # Apply lag of 3
        lagged_info = account_info2.apply_lag(max_lag=3)

        # General asserts
        assert isinstance(lagged_info, dts.AccountInfo)
        assert lagged_info.arr.shape[0] == 3  # 6 original - 3 lag
        assert len(lagged_info.dates) == 3

        # Check array values after lagging
        assert np.allclose(lagged_info.arr, acc_arr2[3:])

        # Check dates mapping after lagging
        expected_date_map = {date: idx for idx, date in enumerate(example_dates[3:])}
        assert lagged_info.dates == expected_date_map

    def test_account_info1_apply_daterange(
        self,
        account_info1: dts.AccountInfo,
        acc_arr1: ArrayF,
        example_dates: list[datetime.date],
    ):
        # Apply date range from 2024-02-01 to 2024-05-01
        start_date = datetime.date(2024, 2, 1)
        end_date = datetime.date(2024, 5, 1)
        filtered_info = account_info1.apply_daterange(
            start_date=start_date,
            end_date=end_date,
        )

        # General asserts
        assert isinstance(filtered_info, dts.AccountInfo)
        assert filtered_info.arr.shape[0] == 4  # Dates from Feb to May
        assert len(filtered_info.dates) == 4

        # Check array values after filtering
        assert np.allclose(filtered_info.arr, acc_arr1[1:5])

        # Check dates mapping after filtering
        expected_date_map = {date: idx for idx, date in enumerate(example_dates[1:5])}
        assert filtered_info.dates == expected_date_map

    def test_account_info2_apply_daterange(
        self,
        account_info2: dts.AccountInfo,
        acc_arr2: ArrayF,
        example_dates: list[datetime.date],
    ):
        # Apply date range from 2024-03-01 to 2024-06-01
        start_date = datetime.date(2024, 3, 1)
        end_date = datetime.date(2024, 6, 1)
        filtered_info = account_info2.apply_daterange(
            start_date=start_date,
            end_date=end_date,
        )

        # General asserts
        assert isinstance(filtered_info, dts.AccountInfo)
        assert filtered_info.arr.shape[0] == 4  # Dates from Mar to Jun
        assert len(filtered_info.dates) == 4

        # Check array values after filtering
        assert np.allclose(filtered_info.arr, acc_arr2[2:6])

        # Check dates mapping after filtering
        expected_date_map = {date: idx for idx, date in enumerate(example_dates[2:6])}
        assert filtered_info.dates == expected_date_map


@pytest.mark.unit
@pytest.mark.datatypes
class TestAccountGroupInfo:
    def test_account_group_info_example(
        self,
        account_group_info_example: dts.AccountGroupInfo,
        acc_arr1: ArrayF,
        acc_arr2: ArrayF,
        acc_arr3: ArrayF,
        example_dates: list[datetime.date],
    ):
        # General asserts
        assert isinstance(account_group_info_example, dts.AccountGroupInfo)
        assert account_group_info_example.arr.shape[0] == 3  # 3 accounts
        assert len(account_group_info_example) == 3
        assert account_group_info_example.arr.shape[1] == 6  # 6 time periods

        assert account_group_info_example.get_ordered_accounts() == [
            dts.AccountType('account_1'),
            dts.AccountType('account_2'),
            dts.AccountType('account_3'),
        ]

        # Segment asserts
        assert account_group_info_example.segment_type.name == dts.ProductType(
            'product_A'
        )

        # Region asserts
        assert account_group_info_example.region_type.name == dts.LocationType(
            'location_X'
        )

        # Check individual account arrays
        acc1 = dts.AccountType('account_1')
        assert acc1 in account_group_info_example.account_map
        acc1_idx = account_group_info_example.account_map[acc1]
        assert acc1_idx == 0
        assert np.allclose(
            account_group_info_example[acc1].arr,
            acc_arr1,
        )
        assert np.allclose(
            account_group_info_example.arr[acc1_idx],
            acc_arr1,
        )

        acc2 = dts.AccountType('account_2')
        assert acc2 in account_group_info_example.account_map
        acc2_idx = account_group_info_example.account_map[acc2]
        assert acc2_idx == 1
        assert np.allclose(
            account_group_info_example[acc2].arr,
            acc_arr2,
        )
        assert np.allclose(
            account_group_info_example.arr[acc2_idx],
            acc_arr2,
        )

        acc3 = dts.AccountType('account_3')
        assert acc3 in account_group_info_example.account_map
        acc3_idx = account_group_info_example.account_map[acc3]
        assert acc3_idx == 2
        assert np.allclose(
            account_group_info_example[acc3].arr,
            acc_arr3,
        )
        assert np.allclose(
            account_group_info_example.arr[acc3_idx],
            acc_arr3,
        )

        # Check dates mapping
        expected_date_map = {date: idx for idx, date in enumerate(example_dates)}
        for date_map in account_group_info_example.dates:
            assert date_map == expected_date_map

    def test_account_group_apply_lag(
        self,
        account_group_info_example: dts.AccountGroupInfo,
        acc_arr1: ArrayF,
        acc_arr2: ArrayF,
        acc_arr3: ArrayF,
        example_dates: list[datetime.date],
    ):
        # Test lagging functionality
        lagged_acc_info = account_group_info_example.apply_lag(max_lag=2)

        assert isinstance(lagged_acc_info, dts.AccountGroupInfo)
        assert lagged_acc_info.arr.shape[0] == 3  # 3 accounts
        assert lagged_acc_info.arr.shape[1] == 4  # 6 original - 2 lag
        assert len(lagged_acc_info) == 3

        # Check that lagging was applied correctly
        acc1 = dts.AccountType('account_1')
        assert np.allclose(
            lagged_acc_info[acc1].arr,
            acc_arr1[2:],
        )

        acc2 = dts.AccountType('account_2')
        assert np.allclose(
            lagged_acc_info[acc2].arr,
            acc_arr2[2:],
        )

        acc3 = dts.AccountType('account_3')
        assert np.allclose(
            lagged_acc_info[acc3].arr,
            acc_arr3[2:],
        )

        # Check dates mapping after lagging
        expected_date_map_lagged = {
            date: idx for idx, date in enumerate(example_dates[2:])
        }
        for date_map in lagged_acc_info.dates:
            assert date_map == expected_date_map_lagged

    def test_account_group_apply_daterange(
        self,
        account_group_info_example: dts.AccountGroupInfo,
        acc_arr1: ArrayF,
        acc_arr2: ArrayF,
        acc_arr3: ArrayF,
        example_dates: list[datetime.date],
    ):
        # Test date range filtering functionality
        start_date = datetime.date(2024, 2, 1)
        end_date = datetime.date(2024, 5, 1)
        filtered_acc_info = account_group_info_example.apply_daterange(
            start_date=start_date,
            end_date=end_date,
        )

        assert isinstance(filtered_acc_info, dts.AccountGroupInfo)
        assert filtered_acc_info.arr.shape[0] == 3  # 3 accounts
        assert filtered_acc_info.arr.shape[1] == 4  # Dates from Jan 2 to Jan 5
        assert len(filtered_acc_info) == 3

        # Check that filtering was applied correctly
        acc1 = dts.AccountType('account_1')
        assert np.allclose(
            filtered_acc_info[acc1].arr,
            acc_arr1[1:5],
        )

        acc2 = dts.AccountType('account_2')
        assert np.allclose(
            filtered_acc_info[acc2].arr,
            acc_arr2[1:5],
        )

        acc3 = dts.AccountType('account_3')
        assert np.allclose(
            filtered_acc_info[acc3].arr,
            acc_arr3[1:5],
        )

        # Check dates mapping after filtering
        expected_date_map_filtered = {
            date: idx for idx, date in enumerate(example_dates[1:5])
        }
        for date_map in filtered_acc_info.dates:
            assert date_map == expected_date_map_filtered

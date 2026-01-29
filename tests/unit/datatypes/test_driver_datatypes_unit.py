import datetime

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

import lh_v2.datatypes as dts
from lh_v2.shared import BASE_NP_DTYPE, ArrayF


@pytest.mark.unit
@pytest.mark.datatypes
class TestDriver:
    def test_driver1(
        self,
        driver1: dts.Driver,
        driv_arr_1: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the Driver fixture 'driver1'."""
        assert driver1.name == dts.DriverName('driver_1')

        assert np.allclose(driver1.arr, driv_arr_1)

        for idx, date in enumerate(example_dates):
            assert driver1.dates[date] == idx

    def test_driver2(
        self,
        driver2: dts.Driver,
        driv_arr_2: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the Driver fixture 'driver2'."""
        assert driver2.name == dts.DriverName('driver_2')

        assert np.allclose(driver2.arr, driv_arr_2)

        for idx, date in enumerate(example_dates):
            assert driver2.dates[date] == idx

    def test_driver3(
        self,
        driver3: dts.Driver,
        driv_arr_3: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the Driver fixture 'driver3'."""
        assert driver3.name == dts.DriverName('driver_3')

        assert np.allclose(driver3.arr, driv_arr_3)

        for idx, date in enumerate(example_dates):
            assert driver3.dates[date] == idx

    def test_driver_apply_daterange(
        self,
        driver1: dts.Driver,
        driv_arr_1: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the apply_date_range method of Driver."""
        # Apply date range to driver1
        start_date = example_dates[2]
        end_date = example_dates[5]

        filtered_driver = driver1.apply_daterange(
            start_date=start_date, end_date=end_date
        )

        # Expected array after applying date range and lagging
        expected_arr = driv_arr_1[2:6]  # Indices 2 to 5 inclusive
        assert np.allclose(filtered_driver.arr, expected_arr)

        # Expected dates mapping after applying date range and lagging
        expected_dates = {date: idx for idx, date in enumerate(example_dates[2:6])}
        assert filtered_driver.dates == expected_dates

        # Ensure original driver is unchanged
        assert np.allclose(driver1.arr, driv_arr_1)
        for idx, date in enumerate(example_dates):
            assert driver1.dates[date] == idx

    def test_driver_add_forecast_vals(
        self,
        driver1: dts.Driver,
        driv_arr_1: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the add_forecast_values method of Driver."""
        # Forecast values to add
        forecast_values = np.array([11.0, 12.0], dtype=BASE_NP_DTYPE)
        forecast_dates = [
            example_dates[-1] + relativedelta(months=1),
            example_dates[-1] + relativedelta(months=2),
        ]

        # Add forecast values to driver1
        extended_driver = driver1.add_forecast_vals(
            forecast_dates=forecast_dates,
            forecast_values=forecast_values,
        )

        # Expected array after adding forecast values
        expected_arr = np.concatenate((driv_arr_1, forecast_values))
        assert np.allclose(extended_driver.arr, expected_arr)

        # Expected dates mapping after adding forecast values
        expected_dates = {
            **{date: idx for idx, date in enumerate(example_dates)},
            **{date: len(driv_arr_1) + idx for idx, date in enumerate(forecast_dates)},
        }
        assert extended_driver.dates == expected_dates

        # Ensure original driver is unchanged
        assert np.allclose(driver1.arr, driv_arr_1)
        for idx, date in enumerate(example_dates):
            assert driver1.dates[date] == idx


@pytest.mark.unit
@pytest.mark.datatypes
class TestDriverGroup:
    def test_driver_group1(
        self,
        driver_group1: dts.DriverGroup,
        driv_arr_1: ArrayF,
        driv_arr_2: ArrayF,
        driv_arr_3: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the DriverGroup fixture 'driver_group1'."""
        expected_driver_names = [
            dts.DriverName('driver_1'),
            dts.DriverName('driver_2'),
            dts.DriverName('driver_3'),
        ]
        assert set(driver_group1.map.keys()) == set(expected_driver_names)
        assert driver_group1.get_ordered_drivers() == expected_driver_names

        expected_arr = np.vstack(
            [driv_arr_1, driv_arr_2, driv_arr_3], dtype=BASE_NP_DTYPE
        )
        assert np.allclose(driver_group1.arr, expected_arr)

        for dates_dict in driver_group1.dates:
            for idx, date in enumerate(example_dates):
                assert dates_dict[date] == idx

        assert np.allclose(driver_group1[0], driv_arr_1)
        assert np.allclose(driver_group1[1], driv_arr_2)
        assert np.allclose(driver_group1[2], driv_arr_3)

        assert np.allclose(
            driver_group1[0:2], np.vstack([driv_arr_1, driv_arr_2], dtype=BASE_NP_DTYPE)
        )

    def test_driver_group2(
        self,
        driver_group2: dts.DriverGroup,
        driv_arr_4: ArrayF,
        driv_arr_5: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the DriverGroup fixture 'driver_group2'."""
        expected_driver_names = [
            dts.DriverName('driver_4'),
            dts.DriverName('driver_5'),
        ]
        assert set(driver_group2.map.keys()) == set(expected_driver_names)
        assert driver_group2.get_ordered_drivers() == expected_driver_names

        expected_arr = np.vstack([driv_arr_4, driv_arr_5], dtype=BASE_NP_DTYPE)
        assert np.allclose(driver_group2.arr, expected_arr)

        for dates_dict in driver_group2.dates:
            for idx, date in enumerate(example_dates):
                assert dates_dict[date] == idx

    def test_from_driver_lst(
        self,
        driver1: dts.Driver,
        driver2: dts.Driver,
        driver3: dts.Driver,
        driver_group1: dts.DriverGroup,
    ):
        """Test the from_driver_lst method of DriverGroup."""
        driver_lst = [driver1, driver2, driver3]
        constructed_group = dts.DriverGroup.from_driver_lst(driver_lst)

        # Check that constructed DriverGroup matches the fixture
        assert constructed_group.map == driver_group1.map
        assert np.allclose(constructed_group.arr, driver_group1.arr)

        for dict1, dict2 in zip(constructed_group.dates, driver_group1.dates):
            assert dict1 == dict2

    def test_apply_lags(
        self,
        driver_group1: dts.DriverGroup,
        driv_arr_1: ArrayF,
        driv_arr_2: ArrayF,
        driv_arr_3: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the apply_lags method of DriverGroup."""
        lags = {
            dts.DriverName('driver_1'): 2,
            dts.DriverName('driver_2'): 1,
            dts.DriverName('driver_3'): 3,
        }
        max_lag = max(lags.values())

        lagged_group = driver_group1.apply_lags(best_lags=lags, max_lag=max_lag)

        # Expected arrays after applying lags
        expected_arr_1 = driv_arr_1[1:-2]  # Lag 2, max_lag 3
        expected_arr_2 = driv_arr_2[2:-1]  # Lag 1, max_lag 3
        expected_arr_3 = driv_arr_3[:-3]  # Lag 3, max_lag 3

        expected_arr = np.vstack(
            [expected_arr_1, expected_arr_2, expected_arr_3], dtype=BASE_NP_DTYPE
        )
        assert np.allclose(lagged_group.arr, expected_arr)

        # Expected dates mapping after applying lags
        dates_lst: list[dict[datetime.date, int]] = []
        for driver in driver_group1.get_ordered_drivers():
            dates_lst.append(
                {
                    date: idx
                    for idx, date in enumerate(
                        example_dates[max_lag - lags[driver] : -lags[driver]]
                    )
                }
            )

        assert lagged_group.dates == dates_lst

    def test_apply_daterange(
        self,
        driver_group1: dts.DriverGroup,
        driv_arr_1: ArrayF,
        driv_arr_2: ArrayF,
        driv_arr_3: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the apply_date_range method of DriverGroup."""
        start_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): example_dates[1],
            dts.DriverName('driver_2'): example_dates[0],
            dts.DriverName('driver_3'): example_dates[2],
        }
        end_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): example_dates[4],
            dts.DriverName('driver_2'): example_dates[3],
            dts.DriverName('driver_3'): example_dates[5],
        }
        filtered_group = driver_group1.apply_daterange(
            start_dates=start_dates, end_dates=end_dates
        )

        # Expected arrays after applying date ranges
        expected_arr_1 = driv_arr_1[1:5]  # Indices 1 to 4 inclusive
        expected_arr_2 = driv_arr_2[0:4]  # Indices 0 to 3 inclusive
        expected_arr_3 = driv_arr_3[2:6]  # Indices 2 to 5 inclusive

        expected_arr = np.vstack(
            [expected_arr_1, expected_arr_2, expected_arr_3], dtype=BASE_NP_DTYPE
        )
        assert np.allclose(filtered_group.arr, expected_arr)

        # Expected dates mapping after applying date ranges
        expected_dates_lst: list[dict[datetime.date, int]] = []
        expected_dates_lst.append(
            {date: idx for idx, date in enumerate(example_dates[1:5])}
        )
        expected_dates_lst.append(
            {date: idx for idx, date in enumerate(example_dates[0:4])}
        )
        expected_dates_lst.append(
            {date: idx for idx, date in enumerate(example_dates[2:6])}
        )

        assert filtered_group.dates == expected_dates_lst

    def test_apply_daterange_invalid_not_all_drivers(
        self,
        driver_group1: dts.DriverGroup,
    ):
        """Test that apply_date_range raises error for invalid date ranges."""
        start_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2020, 1, 1),
        }
        end_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2019, 12, 31),
        }

        with pytest.raises(
            AssertionError,
        ):
            driver_group1.apply_daterange(start_dates=start_dates, end_dates=end_dates)

    def test_apply_daterange_invalid_not_all_driver_end_dates(
        self,
        driver_group1: dts.DriverGroup,
    ):
        """Test that apply_date_range raises error for invalid date ranges."""
        start_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2020, 1, 1),
            dts.DriverName('driver_2'): datetime.date(2020, 1, 1),
            dts.DriverName('driver_3'): datetime.date(2020, 1, 1),
        }
        end_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2020, 12, 31),
            dts.DriverName('driver_2'): datetime.date(2020, 12, 31),
        }

        with pytest.raises(
            AssertionError,
        ):
            driver_group1.apply_daterange(start_dates=start_dates, end_dates=end_dates)

    def test_apply_daterange_invalid_date_not_in_orig_drivergroup(
        self,
        driver_group1: dts.DriverGroup,
    ):
        """Test that apply_date_range raises error for invalid date ranges."""
        start_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2030, 1, 1),
            dts.DriverName('driver_2'): datetime.date(2020, 1, 1),
            dts.DriverName('driver_3'): datetime.date(2020, 1, 1),
        }
        end_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2020, 12, 31),
            dts.DriverName('driver_2'): datetime.date(2020, 12, 31),
            dts.DriverName('driver_3'): datetime.date(2020, 12, 31),
        }

        with pytest.raises(
            ValueError,
        ):
            driver_group1.apply_daterange(start_dates=start_dates, end_dates=end_dates)

    def test_apply_daterange_invalid_different_lengths(
        self,
        driver_group1: dts.DriverGroup,
    ):
        """Test that apply_date_range raises error for invalid date ranges."""
        start_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2024, 2, 1),
            dts.DriverName('driver_2'): datetime.date(2024, 1, 1),
            dts.DriverName('driver_3'): datetime.date(2024, 2, 1),
        }
        end_dates: dict[dts.DriverName, datetime.date] = {
            dts.DriverName('driver_1'): datetime.date(2024, 5, 1),
            dts.DriverName('driver_2'): datetime.date(2024, 4, 1),
            dts.DriverName('driver_3'): datetime.date(2024, 4, 1),
        }

        with pytest.raises(
            ValueError,
        ):
            driver_group1.apply_daterange(start_dates=start_dates, end_dates=end_dates)


@pytest.mark.unit
@pytest.mark.datatypes
class TestClassifiedDriverGroups:
    def test_classified_driver_groups_example(
        self,
        classified_driver_groups_example: dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ],
        driv_arr_1: ArrayF,
        driv_arr_2: ArrayF,
        driv_arr_3: ArrayF,
        driv_arr_4: ArrayF,
        driv_arr_5: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the ClassifiedDriverGroups fixture 'classified_driver_groups_example'."""
        expected_classifications = [
            dts.DriverClassification('class_1'),
            dts.DriverClassification('class_2'),
        ]
        assert set(
            classified_driver_groups_example.classification_groups.keys()
        ) == set(expected_classifications)
        assert (
            classified_driver_groups_example.get_ordered_classifications()
            == expected_classifications
        )

        expected_arr = np.vstack(
            [driv_arr_1, driv_arr_2, driv_arr_3, driv_arr_4, driv_arr_5],
            dtype=BASE_NP_DTYPE,
        )
        assert np.allclose(classified_driver_groups_example.arr, expected_arr)

        for dates_dict in classified_driver_groups_example.dates:
            for idx, date in enumerate(example_dates):
                assert dates_dict[date] == idx

        assert np.allclose(classified_driver_groups_example.arr[0], driv_arr_1)
        assert np.allclose(classified_driver_groups_example.arr[1], driv_arr_2)
        assert np.allclose(classified_driver_groups_example.arr[2], driv_arr_3)
        assert np.allclose(classified_driver_groups_example.arr[3], driv_arr_4)
        assert np.allclose(classified_driver_groups_example.arr[4], driv_arr_5)

    def test_classified_driver_groups_from_lst(
        self,
        classified_driver_groups_example: dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ],
        driver_group1: dts.DriverGroup,
        driver_group2: dts.DriverGroup,
    ):
        """Test the from_drivergroup_lst method of ClassifiedDriverGroups."""
        drivergroup_lst: list[dts.DriverGroup] = [driver_group1, driver_group2]
        classification_groups: dict[dts.DriverClassification, int] = {
            dts.DriverClassification('class_1'): 0,
            dts.DriverClassification('class_2'): 1,
        }
        constructed_cdg = dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ].from_driver_group_lst(
            driver_group_lst=drivergroup_lst,
            classification_groups=classification_groups,
        )

        # Check that constructed ClassifiedDriverGroups matches the fixture
        assert (
            constructed_cdg.classification_groups
            == classified_driver_groups_example.classification_groups
        )
        assert np.allclose(constructed_cdg.arr, classified_driver_groups_example.arr)

        for dict1, dict2 in zip(
            constructed_cdg.dates, classified_driver_groups_example.dates
        ):
            assert dict1 == dict2

    def test_classified_driver_groups_get_drivers_by_classification(
        self,
        classified_driver_groups_example: dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ],
        driver_group1: dts.DriverGroup,
        driver_group2: dts.DriverGroup,
    ):
        """Test the get_drivers_by_classification method of ClassifiedDriverGroups."""
        class_1_drivers = classified_driver_groups_example[
            dts.DriverClassification('class_1')
        ]
        class_2_drivers = classified_driver_groups_example[
            dts.DriverClassification('class_2')
        ]

        assert class_1_drivers.map == driver_group1.map
        assert np.allclose(class_1_drivers.arr, driver_group1.arr)
        assert class_1_drivers.dates == driver_group1.dates

        assert class_2_drivers.map == driver_group2.map
        assert np.allclose(class_2_drivers.arr, driver_group2.arr)
        assert class_2_drivers.dates == driver_group2.dates

    def test_classified_driver_groups_get_ordered_classification_drivers(
        self,
        classified_driver_groups_example: dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ],
    ):
        """Test the get_ordered_drivers method of ClassifiedDriverGroups."""
        expected_ordered_drivers_class_1 = [
            dts.DriverName('driver_1'),
            dts.DriverName('driver_2'),
            dts.DriverName('driver_3'),
        ]
        expected_ordered_drivers_class_2 = [
            dts.DriverName('driver_4'),
            dts.DriverName('driver_5'),
        ]
        assert (
            classified_driver_groups_example.get_ordered_drivers(
                dts.DriverClassification('class_1')
            )
            == expected_ordered_drivers_class_1
        )
        assert (
            classified_driver_groups_example.get_ordered_drivers(
                dts.DriverClassification('class_2')
            )
            == expected_ordered_drivers_class_2
        )

    def test_classified_driver_groups_get_all_ordered_drivers(
        self,
        classified_driver_groups_example: dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ],
    ):
        """Test the get_ordered_drivers method of ClassifiedDriverGroups."""
        expected_ordered_drivers = [
            dts.DriverName('driver_1'),
            dts.DriverName('driver_2'),
            dts.DriverName('driver_3'),
            dts.DriverName('driver_4'),
            dts.DriverName('driver_5'),
        ]
        assert (
            classified_driver_groups_example.get_all_ordered_drivers()
            == expected_ordered_drivers
        )

    def test_classified_driver_groups_apply_lag(
        self,
        classified_driver_groups_example: dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ],
        driv_arr_1: ArrayF,
        driv_arr_2: ArrayF,
        driv_arr_3: ArrayF,
        driv_arr_4: ArrayF,
        driv_arr_5: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the apply_lags method of ClassifiedDriverGroups."""
        class1 = dts.DriverClassification('class_1')
        class2 = dts.DriverClassification('class_2')
        lags: dict[dts.DriverClassification, dict[dts.DriverName, int]] = {
            class1: {
                dts.DriverName('driver_1'): 2,
                dts.DriverName('driver_2'): 1,
                dts.DriverName('driver_3'): 3,
            },
            class2: {
                dts.DriverName('driver_4'): 0,
                dts.DriverName('driver_5'): 2,
            },
        }
        max_lag = max(
            lag for class_lags in lags.values() for lag in class_lags.values()
        )

        lagged_cdg = classified_driver_groups_example.apply_lag(
            best_lags=lags, max_lag=max_lag
        )

        # Expected arrays after applying lags
        expected_arr_1 = driv_arr_1[1:-2]  # Lag 2, max_lag 3
        expected_arr_2 = driv_arr_2[2:-1]  # Lag 1, max_lag 3
        expected_arr_3 = driv_arr_3[:-3]  # Lag 3, max_lag 3
        expected_arr_4 = driv_arr_4[3:]  # Lag 0, max_lag 3
        expected_arr_5 = driv_arr_5[1:-2]  # Lag 2, max_lag 3

        expected_arr = np.vstack(
            [
                expected_arr_1,
                expected_arr_2,
                expected_arr_3,
                expected_arr_4,
                expected_arr_5,
            ],
            dtype=BASE_NP_DTYPE,
        )
        assert np.allclose(lagged_cdg.arr, expected_arr)

        # Expected dates mapping after applying lags
        dates_lst: list[dict[datetime.date, int]] = []
        for class_ in classified_driver_groups_example.get_ordered_classifications():
            lags_class = lags[class_]
            for driver in classified_driver_groups_example.get_ordered_drivers(class_):
                dates_lst.append(
                    {
                        date: idx
                        for idx, date in enumerate(
                            example_dates[
                                max_lag - lags_class[driver] : -lags_class[driver]
                                if lags_class[driver] != 0
                                else None
                            ]
                        )
                    }
                )

        print(f'EXPECTED DATES LIST: {dates_lst}\n')  # Debug print
        print(f'LAGGED CDG DATES: {lagged_cdg.dates}')  # Debug print

        assert lagged_cdg.dates == dates_lst

    def test_classified_driver_groups_apply_daterange(
        self,
        classified_driver_groups_example: dts.ClassifiedDriverGroups[
            dts.DriverClassification
        ],
        driv_arr_1: ArrayF,
        driv_arr_2: ArrayF,
        driv_arr_3: ArrayF,
        driv_arr_4: ArrayF,
        driv_arr_5: ArrayF,
        example_dates: list[datetime.date],
    ):
        """Test the apply_date_range method of ClassifiedDriverGroups."""
        start_date = example_dates[1]
        end_date = example_dates[4]
        filtered_cdg = classified_driver_groups_example.apply_daterange(
            start_date=start_date, end_date=end_date
        )

        # Expected arrays after applying date ranges
        expected_arr_1 = driv_arr_1[1:5]  # Indices 1 to 4 inclusive
        expected_arr_2 = driv_arr_2[1:5]  # Indices 1 to 4 inclusive
        expected_arr_3 = driv_arr_3[1:5]  # Indices 1 to 4 inclusive
        expected_arr_4 = driv_arr_4[1:5]  # Indices 1 to 4 inclusive
        expected_arr_5 = driv_arr_5[1:5]  # Indices 1 to 4 inclusive

        expected_arr = np.vstack(
            [
                expected_arr_1,
                expected_arr_2,
                expected_arr_3,
                expected_arr_4,
                expected_arr_5,
            ],
            dtype=BASE_NP_DTYPE,
        )
        assert np.allclose(filtered_cdg.arr, expected_arr)

        # Expected dates mapping after applying date ranges
        expected_dates_lst: list[dict[datetime.date, int]] = []
        for _ in range(filtered_cdg.arr.shape[0]):
            expected_dates_lst.append(
                {date: idx for idx, date in enumerate(example_dates[1:5])}
            )

        assert filtered_cdg.dates == expected_dates_lst

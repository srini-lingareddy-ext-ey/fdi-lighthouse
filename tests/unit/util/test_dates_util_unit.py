import datetime

import pytest

from lh_v2.util.dates_util import month_dif, ymd2pydate


@pytest.mark.unit
@pytest.mark.util
class TestDatesUtil:
    def test_month_dif(self):
        # Test cases for month_dif function
        date1 = datetime.date(2023, 1, 1)
        date2 = datetime.date(2023, 4, 1)
        assert month_dif(date1, date2) == 3

        date3 = datetime.date(2022, 12, 1)
        date4 = datetime.date(2023, 1, 1)
        assert month_dif(date3, date4) == 1

        date5 = datetime.date(2020, 5, 1)
        date6 = datetime.date(2023, 5, 1)
        assert month_dif(date5, date6) == 36

        date7 = datetime.date(2021, 6, 1)
        date8 = datetime.date(2021, 6, 1)
        assert month_dif(date7, date8) == 0

        date9 = datetime.date(2021, 11, 1)
        date10 = datetime.date(2021, 2, 1)
        assert month_dif(date9, date10) == -9

        date11 = datetime.date(2021, 3, 1)
        date12 = datetime.date(2019, 4, 1)
        assert month_dif(date11, date12) == -23

    def test_ymd2pydate(self):
        # Test cases for ymd2pydate function
        date_str1 = '2023-01-15'
        expected_date1 = datetime.date(2023, 1, 15)
        assert ymd2pydate(date_str1) == expected_date1

        date_str2 = '2020-12-31'
        expected_date2 = datetime.date(2020, 12, 31)
        assert ymd2pydate(date_str2) == expected_date2

        date_str3 = '1999-07-04'
        expected_date3 = datetime.date(1999, 7, 4)
        assert ymd2pydate(date_str3) == expected_date3

        date_str4 = '2000-02-29'  # Leap year
        expected_date4 = datetime.date(2000, 2, 29)
        assert ymd2pydate(date_str4) == expected_date4

        date_str5 = '2021-11-01'
        expected_date5 = datetime.date(2021, 11, 1)
        assert ymd2pydate(date_str5) == expected_date5

import datetime
from enum import Enum

import numpy as np
from dateutil.relativedelta import relativedelta


def ymd2npdate(ymd_str: str) -> np.datetime64:
    ymd_split = ymd_str.split('-')
    y = int(ymd_split[0])
    m = int(ymd_split[1])
    d = int(ymd_split[2])
    return np.datetime64(datetime.date(year=y, month=m, day=d))


def ymd2pydate(ymd_str: str) -> datetime.date:
    ymd_split = ymd_str.split('-')
    y = int(ymd_split[0])
    m = int(ymd_split[1])
    d = int(ymd_split[2])
    return datetime.date(year=y, month=m, day=d)


def npdate_add_months(date: np.datetime64, n_months: int) -> np.datetime64:
    delta = np.timedelta64(n_months, 'M')
    return (date.astype('datetime64[M]') + delta).astype('datetime64[D]')


def month_dif(start_date: datetime.date, end_date: datetime.date) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


class DateFrequencyEnum(str, Enum):
    DAILY = 'D'
    WEEKLY = 'W'
    MONTHLY = 'M'
    QUARTERLY = 'Q'
    YEARLY = 'Y'


class Datetime:
    date: datetime.date
    freq: DateFrequencyEnum


def time_dif(start_date: Datetime, end_date: Datetime) -> int:
    if start_date.freq != end_date.freq:
        raise ValueError('Frequency mismatch between start_date and end_date')
    match start_date.freq:
        case DateFrequencyEnum.DAILY:
            delta = end_date.date - start_date.date
            return delta.days
        case DateFrequencyEnum.WEEKLY:
            delta = end_date.date - start_date.date
            return delta.days // 7
        case DateFrequencyEnum.MONTHLY:
            return month_dif(start_date.date, end_date.date)
        case DateFrequencyEnum.QUARTERLY:
            return month_dif(start_date.date, end_date.date) // 3
        case DateFrequencyEnum.YEARLY:
            return end_date.date.year - start_date.date.year
        case _:
            raise ValueError(f'Unsupported frequency: {start_date.freq}')


def add_time_unit(start_date: Datetime, n_units: int) -> datetime.date:
    match start_date.freq:
        case DateFrequencyEnum.DAILY:
            return start_date.date + datetime.timedelta(days=n_units)
        case DateFrequencyEnum.WEEKLY:
            return start_date.date + datetime.timedelta(weeks=n_units)
        case DateFrequencyEnum.MONTHLY:
            return start_date.date + relativedelta(months=n_units)
        case DateFrequencyEnum.QUARTERLY:
            return start_date.date + relativedelta(months=3 * n_units)
        case DateFrequencyEnum.YEARLY:
            return start_date.date + relativedelta(years=n_units)
        case _:
            raise ValueError(f'Unsupported frequency: {start_date.freq}')

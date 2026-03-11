import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

import lh_v2.datatypes as dts
from lh_v2.shared import ArrayF

TimeSeries = TypeVar('TimeSeries', bound=dts.Driver | dts.AccountInfo)


def _get_historicals(
    timeseries: dts.Driver | dts.AccountInfo,
    forecast_daterange: tuple[datetime.date, datetime.date],
) -> tuple[list[datetime.date], ArrayF]:
    historical_dates = [
        date for date in timeseries.dates.keys() if date < forecast_daterange[0]
    ]
    historical_values = timeseries.apply_daterange(end_date=historical_dates[-1]).arr
    return historical_dates, historical_values


def get_historicals_forecast(
    timeseries_base: dts.Driver | dts.AccountInfo,
    forecast_daterange: tuple[datetime.date, datetime.date],
) -> ArrayF:
    # Get the driver values for the forecast period
    forecast_values = timeseries_base.apply_daterange(
        start_date=forecast_daterange[0], end_date=forecast_daterange[1]
    ).arr

    return forecast_values


def format_timeseries_for_plotting(
    timeseries: dts.Driver | dts.AccountInfo,
    forecast_daterange: tuple[datetime.date, datetime.date],
    timeseries_base: dts.Driver | dts.AccountInfo | None = None,
) -> tuple[
    list[datetime.date],
    ArrayF,
    list[datetime.date],
    ArrayF,
    ArrayF | None,
]:
    historical_dates, historical_values = _get_historicals(
        timeseries=timeseries,
        forecast_daterange=forecast_daterange,
    )

    # Extract all dates within the forecast date range (inclusive)
    forecast_dates = [
        date
        for date in timeseries.dates.keys()
        if forecast_daterange[0] <= date <= forecast_daterange[1]
    ]

    # Get the driver values for the forecast period
    forecast_values = timeseries.apply_daterange(
        start_date=forecast_daterange[0], end_date=forecast_daterange[1]
    ).arr

    if timeseries_base is not None:
        forecast_base_values = get_historicals_forecast(
            timeseries_base, forecast_daterange
        )
    else:
        forecast_base_values = None

    return (
        historical_dates,
        historical_values,
        forecast_dates,
        forecast_values,
        forecast_base_values,
    )


def format_timeseries_mapping_for_plotting(
    timeseries_mapping: Mapping[Any, dts.Driver] | Mapping[Any, dts.AccountInfo],
    forecast_daterange: tuple[datetime.date, datetime.date],
    timeseries_base: dts.Driver | dts.AccountInfo | None = None,
) -> tuple[
    list[datetime.date],
    ArrayF,
    list[datetime.date],
    dict[Any, ArrayF],
    ArrayF | None,
]:
    mappings = list(timeseries_mapping.keys())
    base_dates = set(timeseries_mapping[mappings[0]].dates.keys())
    for mapping in mappings[1:]:
        assert set(timeseries_mapping[mapping].dates.keys()) == base_dates, (
            'All timeseries in the mapping must have the same dates for plotting.'
        )

    historical_dates, historical_values = _get_historicals(
        timeseries=timeseries_mapping[mappings[0]],
        forecast_daterange=forecast_daterange,
    )

    forecast_dates = [
        date
        for date in timeseries_mapping[mappings[0]].dates.keys()
        if forecast_daterange[0] <= date <= forecast_daterange[1]
    ]

    forecast_values: dict[Any, ArrayF] = {
        mapping: timeseries_mapping[mapping]
        .apply_daterange(
            start_date=forecast_daterange[0], end_date=forecast_daterange[1]
        )
        .arr
        for mapping in mappings
    }

    if timeseries_base is not None:
        forecast_base_values = get_historicals_forecast(
            timeseries_base, forecast_daterange
        )
    else:
        forecast_base_values = None

    return (
        historical_dates,
        historical_values,
        forecast_dates,
        forecast_values,
        forecast_base_values,
    )

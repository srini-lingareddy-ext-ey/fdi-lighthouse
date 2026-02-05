import datetime
import pathlib as pth

import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

from lh_v2.datatypes import AccountGroupInfo, LocationType, ProductType
from lh_v2.driver_analysis import DriverAnalysisOutput
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingOutput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingOutput,
)
from lh_v2.util import month_dif


def save_selected_drivers_csv(
    da_output: DriverAnalysisOutput,
    segment_type: ProductType,
    region_type: LocationType,
    output_pth: pth.Path,
):
    out_dict: dict[str, list[str | int]] = {
        'account_type': [],
        'segment_type': [],
        'region_type': [],
        'classification': [],
        'driver_name': [],
        'lag': [],
    }
    for acc in da_output.selected_drivers.keys():
        for classification in da_output.selected_drivers[acc].keys():
            for driver in da_output.selected_drivers[acc][classification]:
                out_dict['account_type'].append(str(acc))
                out_dict['segment_type'].append(str(segment_type))
                out_dict['region_type'].append(str(region_type))
                out_dict['classification'].append(str(classification))
                out_dict['driver_name'].append(str(driver))
                out_dict['lag'].append(da_output.lags[acc][classification][driver])

    out_df = pl.DataFrame(out_dict)
    out_df.write_csv(output_pth.with_suffix('.csv'))
    return


def save_account_validation_csv(
    validation_output: ModelTrainingOutput,
    actuals: AccountGroupInfo,
    validation_daterange: tuple[datetime.date, datetime.date],
    output_pth: pth.Path,
):
    assert set(validation_output.account_map.keys()) == set(
        actuals.get_ordered_accounts()
    )

    out_dict: dict[str, list[datetime.date | str | float]] = {
        'date': [],
        'account_key': [],
        'segment': [],
        'region': [],
        'method': [],
        'value': [],
    }

    actuals_filtered = actuals.apply_daterange(
        start_date=validation_daterange[0],
        end_date=validation_daterange[1],
    )

    dates: list[datetime.date] = []
    for i in range(
        month_dif(
            validation_daterange[0],
            validation_daterange[1],
        )
        + 1
    ):
        dates.append(validation_daterange[0] + relativedelta(months=i))

    for acc in validation_output.account_map.keys():
        acc_idx = validation_output.account_map[acc]
        for method in validation_output.method_map[acc].keys():
            method_idx = validation_output.method_map[acc][method]
            for date, val in zip(
                dates, validation_output.forecasted_accounts[acc_idx][method_idx].arr
            ):
                out_dict['date'].append(date)
                out_dict['account_key'].append(str(acc))
                out_dict['segment'].append(str(actuals.segment_type.name))
                out_dict['region'].append(str(actuals.region_type.name))
                out_dict['method'].append(method.value)
                out_dict['value'].append(val)
        for date, val in zip(
            dates, actuals_filtered.arr[actuals_filtered.account_map[acc]]
        ):
            out_dict['date'].append(date)
            out_dict['account_key'].append(str(acc))
            out_dict['segment'].append(str(actuals.segment_type.name))
            out_dict['region'].append(str(actuals.region_type.name))
            out_dict['method'].append('actual')
            out_dict['value'].append(val)

    out_df = pl.DataFrame(out_dict)
    out_df.write_csv(output_pth.with_suffix('.csv'))
    return


def save_account_forecasts_csv(
    account_forecasts: ModelForecastingOutput, output_pth: pth.Path
):
    dates: list[datetime.date] = []
    for i in range(
        month_dif(
            account_forecasts.forecast_daterange[0],
            account_forecasts.forecast_daterange[1],
        )
        + 1
    ):
        dates.append(account_forecasts.forecast_daterange[0] + relativedelta(months=i))
    out_dict: dict[str, list[datetime.date] | np.ndarray] = {'date': dates}

    account_forecasts_filtered = account_forecasts.accounts_forecasts.apply_daterange(
        start_date=account_forecasts.forecast_daterange[0],
        end_date=account_forecasts.forecast_daterange[1],
    )

    for acc in account_forecasts_filtered.get_ordered_accounts():
        out_dict[str(acc)] = account_forecasts_filtered.arr[
            account_forecasts_filtered.account_map[acc]
        ]

    out_df = pl.DataFrame(out_dict)

    out_df = out_df.unpivot(
        index='date',
        variable_name='account_key',
        value_name='forecast_value',
    )

    out_df = out_df.with_columns(
        pl.lit(str(account_forecasts.accounts_forecasts.segment_type.name)).alias(
            'segment_type'
        )
    )
    out_df = out_df.with_columns(
        pl.lit(str(account_forecasts.accounts_forecasts.region_type.name)).alias(
            'region_type'
        )
    )

    out_df.write_csv(output_pth.with_suffix('.csv'))
    return

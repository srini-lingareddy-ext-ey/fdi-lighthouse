import datetime
from typing import Literal

import polars as pl

import lh_v2.datatypes as dts
from lh_v2.params import OutputParams
from lh_v2.util import create_output_dir, get_logger, get_output_dir

logger = get_logger(__name__)


def save_driver_forecasts(
    driver_forecasts: dict[dts.AccountType, dts.DriverGroup],
    driver_lags: dict[dts.AccountType, dict[dts.DriverName, int]],
    classification_map: dict[
        dts.AccountType, dict[dts.DriverName, dts.DriverClassification]
    ],
    forecasting_daterange: tuple[datetime.date, datetime.date],
    forecasting_type: Literal['validation', 'forecast'],
    output_params: OutputParams,
):
    output_file_name = f'driver_forecasts_{forecasting_type}'

    if not output_params.b_save_info:
        return

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    accounts: list[dts.AccountType] = []
    driver_names: list[dts.DriverName] = []
    classifications: list[dts.DriverClassification] = []
    lags: list[int] = []
    forecast_values: list[list[float]] = []

    for account in driver_forecasts.keys():
        driver_forecast = driver_forecasts[account].apply_daterange_lags(
            start_date=forecasting_daterange[0],
            end_date=forecasting_daterange[1],
            lags=driver_lags[account],
            b_training=False,
        )
        for driver in driver_forecast.get_ordered_drivers():
            accounts.append(account)
            driver_names.append(driver)
            classifications.append(classification_map[account][driver])
            lags.append(driver_lags[account][driver])
            forecast_values.append([float(val) for val in driver_forecast[driver]])

    pivoted_forecast_values: list[list[float]] = []
    for idx2 in range(len(forecast_values[0])):
        pivoted_forecast_values.append([])
        for idx1 in range(len(forecast_values)):
            pivoted_forecast_values[idx2].append(forecast_values[idx1][idx2])

    dict_out: dict[
        str,
        list[dts.AccountType]
        | list[dts.DriverName]
        | list[dts.DriverClassification]
        | list[int]
        | list[float],
    ] = {
        'account': accounts,
        'driver_name': driver_names,
        'classification': classifications,
        'lag': lags,
    }
    for i in range(len(pivoted_forecast_values)):
        dict_out[f'forecast_val_{i + 1}'] = pivoted_forecast_values[i]

    df_out = pl.DataFrame(dict_out)
    if output_params.table_file_format == 'csv':
        df_out.write_csv(
            get_output_dir(sub_dir_name=output_params.get_save_dir_name())
            / f'{output_file_name}.csv'
        )
    elif output_params.table_file_format == 'parquet':
        df_out.write_parquet(
            get_output_dir(sub_dir_name=output_params.get_save_dir_name())
            / f'{output_file_name}.parquet'
        )
    else:
        raise ValueError(
            f'Unsupported file format: {output_params.table_file_format}. Supported formats are "csv" and "parquet".'
        )

    logger.info(
        f'Saved driver forecasts for forecasting type "{forecasting_type}" to {output_params.table_file_format} '
        f'file in directory: {output_params.get_save_dir_name()}'
    )
    return

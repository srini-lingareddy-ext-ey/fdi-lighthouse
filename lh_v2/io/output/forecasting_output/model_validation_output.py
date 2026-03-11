import datetime
import json
from typing import Any

import polars as pl

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.params import AccountForecastParams, OutputParams
from lh_v2.util import create_output_dir, get_logger, get_output_dir

logger = get_logger(__name__)


def save_model_validation_forecasts(
    account_map: dict[dts.AccountType, int],
    method_map: dict[dts.AccountType, dict[aft.AccountForecastingMethodEnum, int]],
    account_forecasts: list[list[dts.AccountInfo]],
    metrics: list[list[dict[aft.AccountValidationMetricEnum, float]]],
    forecast_daterange: tuple[datetime.date, datetime.date],
    output_params: OutputParams,
):
    output_file_name = 'account_forecasts_validation'
    if not output_params.b_save_info:
        return

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    val_metrics: list[aft.AccountValidationMetricEnum] = list(metrics[0][0].keys())

    accounts: list[dts.AccountType] = []
    methods: list[aft.AccountForecastingMethodEnum] = []
    forecast_values: list[list[float]] = []
    metric_values: list[list[float]] = []

    for account, acc_idx in account_map.items():
        for method, method_idx in method_map[account].items():
            filtered_acc = account_forecasts[acc_idx][method_idx].apply_daterange(
                start_date=forecast_daterange[0], end_date=forecast_daterange[1]
            )

            accounts.append(account)
            methods.append(method)
            forecast_values.append([float(val) for val in filtered_acc.arr])
            metric_lst: list[float] = []
            for metric in val_metrics:
                metric_lst.append(metrics[acc_idx][method_idx][metric])
            metric_values.append(metric_lst)

    pivoted_forecast_values: list[list[float]] = []
    for idx2 in range(len(forecast_values[0])):
        pivoted_forecast_values.append([])
        for idx1 in range(len(forecast_values)):
            pivoted_forecast_values[idx2].append(forecast_values[idx1][idx2])

    dict_out: dict[
        str,
        list[dts.AccountType] | list[aft.AccountForecastingMethodEnum] | list[float],
    ] = {
        'account': accounts,
        'method': methods,
    }

    for idx, metric in enumerate(val_metrics):
        dict_out[metric] = [metric_values[i][idx] for i in range(len(metric_values))]

    for i in range(len(pivoted_forecast_values)):
        dict_out[f'forecast_val_{i}'] = pivoted_forecast_values[i]

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
        f'Saved account forecasting validation results to {output_params.table_file_format} file in directory: '
        f'{output_params.get_save_dir_name()}'
    )

    return


def save_model_validation_params(
    account_map: dict[dts.AccountType, int],
    best_params: list[AccountForecastParams],
    output_params: OutputParams,
):
    output_file_name = 'model_validation_params'

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    dict_out: dict[dts.AccountType, Any] = {}
    for account, acc_idx in account_map.items():
        dict_out[account] = best_params[acc_idx].model_dump(mode='json')

    if output_params.other_file_format == 'json':
        with open(
            get_output_dir(sub_dir_name=output_params.get_save_dir_name())
            / f'{output_file_name}.json',
            'w',
        ) as f:
            json.dump(dict_out, f, indent=4)
    else:
        raise ValueError(
            f'Unsupported file format: {output_params.other_file_format}. Supported formats are "json".'
        )

    logger.info(
        f'Saved account forecasting validation parameters to {output_params.other_file_format} file in directory: '
        f'{output_params.get_save_dir_name()}'
    )

    return

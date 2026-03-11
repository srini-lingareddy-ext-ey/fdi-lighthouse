import datetime

import polars as pl

import lh_v2.datatypes as dts
from lh_v2.params import OutputParams
from lh_v2.util import create_output_dir, get_logger, get_output_dir

logger = get_logger(__name__)


def save_account_reconciliation_forecasts(
    account_forecasts: dts.AccountGroupInfo,
    forecast_daterange: tuple[datetime.date, datetime.date],
    output_params: OutputParams,
):
    output_file_name = 'account_forecasts_reconciled'
    if not output_params.b_save_info:
        return

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    accounts: list[dts.AccountType] = []
    forecast_values: list[list[float]] = []

    filt_accounts = account_forecasts.apply_daterange(
        start_date=forecast_daterange[0], end_date=forecast_daterange[1]
    )
    for account in account_forecasts.get_ordered_accounts():
        accounts.append(account)
        forecast_values.append(
            [
                float(val)
                for val in filt_accounts.arr[filt_accounts.account_map[account]]
            ]
        )

    pivoted_forecast_values: list[list[float]] = []
    for idx2 in range(len(forecast_values[0])):
        pivoted_forecast_values.append([])
        for idx1 in range(len(forecast_values)):
            pivoted_forecast_values[idx2].append(forecast_values[idx1][idx2])

    dict_out: dict[str, list[dts.AccountType] | list[float]] = {
        'account': accounts,
    }
    for idx in range(len(pivoted_forecast_values)):
        dict_out[f'forecast_val_{idx}'] = pivoted_forecast_values[idx]

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
            f"Unsupported file format: {output_params.table_file_format}. Supported formats are 'csv' and 'parquet'."
        )

    logger.info(
        f'Saved reconciled account forecasts to {output_params.table_file_format} '
        f'file at {get_output_dir(sub_dir_name=output_params.get_save_dir_name()) / output_file_name}'
    )

    return

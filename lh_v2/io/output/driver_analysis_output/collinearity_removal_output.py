from typing import Sequence

import polars as pl

import lh_v2.datatypes as dts
from lh_v2.params import OutputParams
from lh_v2.shared import ArrayF
from lh_v2.util import create_output_dir, get_logger, get_output_dir

logger = get_logger(__name__)


def save_collinearity_results(
    dict_collinearity_arr: dict[dts.AccountType, ArrayF],
    ordered_drivers: Sequence[dts.DriverName],
    output_params: OutputParams,
):
    output_file_name = 'collinearity_arr'
    if not output_params.b_save_info:
        return

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    accounts: list[dts.AccountType] = []
    driver1: list[dts.DriverName] = []
    driver2: list[dts.DriverName] = []
    collinearity_values: list[float] = []

    for account in dict_collinearity_arr.keys():
        for idx1 in range(len(ordered_drivers)):
            for idx2 in range(idx1 + 1, len(ordered_drivers)):
                accounts.append(account)
                driver1.append(ordered_drivers[idx1])
                driver2.append(ordered_drivers[idx2])
                collinearity_values.append(dict_collinearity_arr[account][idx1, idx2])

    out_dict: dict[
        str,
        list[dts.AccountType] | list[dts.DriverName] | list[float],
    ] = {
        'account': accounts,
        'driver1': driver1,
        'driver2': driver2,
        'collinearity_value': collinearity_values,
    }

    out_df = pl.DataFrame(out_dict)
    if output_params.table_file_format == 'csv':
        out_df.write_csv(
            get_output_dir(sub_dir_name=output_params.get_save_dir_name())
            / f'{output_file_name}.csv'
        )
    elif output_params.table_file_format == 'parquet':
        out_df.write_parquet(
            get_output_dir(sub_dir_name=output_params.get_save_dir_name())
            / f'{output_file_name}.parquet'
        )
    else:
        raise ValueError(
            f'Unsupported file format: {output_params.table_file_format}. '
            'Supported formats are: csv, parquet.'
        )
    return

import polars as pl

import lh_v2.datatypes as dts
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingMetric
from lh_v2.params import LagParams, OutputParams
from lh_v2.util import create_output_dir, get_logger, get_output_dir

logger = get_logger(__name__)


def save_lag_ranking_results(
    lag_ranking_results: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[DriverRankingMetric, float]],
        ],
    ],
    classification_map: dict[dts.DriverName, dts.DriverClassification],
    best_lags: dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[dts.DriverName, int]],
    ],
    lag_params: LagParams,
    output_params: OutputParams,
):
    output_file_name = 'lag_ranking_results'
    if not output_params.b_save_info:
        return

    if not lag_params.b_lag or lag_params.n_max_lag == 0:
        return

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    acc0 = next(iter(lag_ranking_results.keys()))
    driver0 = next(iter(lag_ranking_results[acc0].keys()))
    lag0 = next(iter(lag_ranking_results[acc0][driver0].keys()))
    metrics = list(lag_ranking_results[acc0][driver0][lag0].keys())

    accounts: list[dts.AccountType] = []
    driver_names: list[dts.DriverName] = []
    lag_values: list[str] = []
    selected_lags: list[int] = []
    dict_metrics: dict[DriverRankingMetric, list[float]] = {
        metric: [] for metric in metrics
    }

    for account in lag_ranking_results.keys():
        for driver_name in lag_ranking_results[account].keys():
            for lag_value in lag_ranking_results[account][driver_name].keys():
                accounts.append(account)
                driver_names.append(dts.DriverName(driver_name))
                selected_lags.append(
                    best_lags[account][classification_map[dts.DriverName(driver_name)]][
                        dts.DriverName(driver_name)
                    ]
                )
                lag_values.append(lag_value)
                for metric in metrics:
                    dict_metrics[metric].append(
                        lag_ranking_results[account][driver_name][lag_value][metric]
                    )

    out_dict: dict[
        str,
        list[dts.AccountType]
        | list[dts.DriverClassification]
        | list[dts.DriverName]
        | list[int]
        | list[float]
        | list[str],
    ] = {
        'account': accounts,
        'driver_name': driver_names,
        'selected_lag': selected_lags,
        'lag_value': lag_values,
    }
    for metric in metrics:
        out_dict[metric.value] = dict_metrics[metric]

    df = pl.DataFrame(out_dict)
    if output_params.table_file_format == 'csv':
        df.write_csv(
            get_output_dir(sub_dir_name=output_params.get_save_dir_name())
            / f'{output_file_name}.csv'
        )
    elif output_params.table_file_format == 'parquet':
        df.write_parquet(
            get_output_dir(sub_dir_name=output_params.get_save_dir_name())
            / f'{output_file_name}.parquet'
        )
    else:
        raise ValueError(
            f'Unsupported file format: {output_params.table_file_format}. '
            'Supported formats are: csv, parquet.'
        )

    logger.info(
        f'Lag ranking results saved to {output_params.get_save_dir_name()} '
        f'with format {output_params.table_file_format}.'
    )

    return

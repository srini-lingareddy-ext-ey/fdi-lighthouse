import polars as pl

import lh_v2.datatypes as dts
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingMetric
from lh_v2.params import OutputParams
from lh_v2.util import create_output_dir, get_logger, get_output_dir

logger = get_logger(__name__)


def save_driver_ranking_results(
    driver_ranking_results: dict[
        dts.AccountType,
        dict[
            dts.DriverClassification,
            dict[dts.DriverName, dict[DriverRankingMetric, float]],
        ],
    ],
    best_lags: dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[dts.DriverName, int]],
    ],
    output_params: OutputParams,
):
    output_file_name = 'driver_ranking_results'
    if not output_params.b_save_info:
        return

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    acc0 = next(iter(driver_ranking_results.keys()))
    class0 = next(iter(driver_ranking_results[acc0].keys()))
    driver0 = next(iter(driver_ranking_results[acc0][class0].keys()))
    metrics = list(driver_ranking_results[acc0][class0][driver0].keys())

    accounts: list[dts.AccountType] = []
    classifications: list[dts.DriverClassification] = []
    driver_names: list[dts.DriverName] = []
    lags: list[int] = []
    dict_metrics: dict[DriverRankingMetric, list[float]] = {
        metric: [] for metric in metrics
    }

    for account in driver_ranking_results.keys():
        for class_ in driver_ranking_results[account].keys():
            for driver in driver_ranking_results[account][class_].keys():
                accounts.append(account)
                classifications.append(class_)
                driver_names.append(driver)
                lags.append(best_lags[account][class_][driver])
                for metric in metrics:
                    dict_metrics[metric].append(
                        driver_ranking_results[account][class_][driver][metric]
                    )

    out_dict: dict[
        str,
        list[dts.AccountType]
        | list[dts.DriverClassification]
        | list[dts.DriverName]
        | list[int]
        | list[float],
    ] = {
        'account': accounts,
        'classification': classifications,
        'driver_name': driver_names,
        'lag': lags,
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
        f'Driver ranking results saved to {output_params.get_save_dir_name()} '
        f'with format {output_params.table_file_format}.'
    )

    return

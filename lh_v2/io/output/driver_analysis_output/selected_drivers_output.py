import polars as pl

import lh_v2.datatypes as dts
from lh_v2.params import OutputParams
from lh_v2.util import create_output_dir, get_logger, get_output_dir

logger = get_logger(__name__)


def save_selected_drivers_results(
    driver_rankings: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ],
    selected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ],
    skipped_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ],
    output_params: OutputParams,
):
    output_file_name = 'selected_drivers'
    if not output_params.b_save_info:
        return

    create_output_dir(sub_dir_name=output_params.get_save_dir_name())

    accounts: list[dts.AccountType] = []
    classifications: list[dts.DriverClassification] = []
    driver_names: list[dts.DriverName] = []
    ranking: list[int] = []
    statuses: list[str] = []

    ordered_drivers_ranking: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}
    for account in driver_rankings.keys():
        ordered_drivers_ranking[account] = {}
        for class_ in driver_rankings[account].keys():
            ordered_drivers_ranking[account][class_] = sorted(
                driver_rankings[account][class_].keys(),
                key=lambda driver: driver_rankings[account][class_][driver],
            )

    for account in ordered_drivers_ranking.keys():
        for class_ in ordered_drivers_ranking[account].keys():
            for driver in ordered_drivers_ranking[account][class_]:
                accounts.append(account)
                classifications.append(class_)
                driver_names.append(driver)
                ranking.append(driver_rankings[account][class_][driver])

                if driver in selected_drivers[account][class_]:
                    statuses.append('selected')
                elif driver in skipped_drivers[account][class_]:
                    statuses.append('skipped')
                else:
                    statuses.append('not_considered')

    out_dict: dict[
        str,
        list[dts.AccountType]
        | list[dts.DriverClassification]
        | list[dts.DriverName]
        | list[str]
        | list[int],
    ] = {
        'account': accounts,
        'classification': classifications,
        'driver_name': driver_names,
        'ranking': ranking,
        'status': statuses,
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

    logger.info(
        f'Selected drivers results saved to {output_params.get_save_dir_name()} '
        f'with format {output_params.table_file_format}.'
    )

    return

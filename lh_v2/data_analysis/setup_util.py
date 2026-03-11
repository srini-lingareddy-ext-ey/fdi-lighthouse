import pathlib as pth

import lh_v2.datatypes as dts
from lh_v2.datatypes.data_loader_types import DataLoadingMethodEnum
from lh_v2.driver_analysis import DriverAnalysisInput, analyze_drivers_full
from lh_v2.io.data_loading.load_data import load_data_driver_ranking
from lh_v2.params import (
    DriverAnalysisParams,
    GeneralParams,
    LighthouseParams,
    OutputParams,
    parse_yaml,
)

path = pth.Path(__file__)


def load_data_data_analysis() -> tuple[
    LighthouseParams, dts.AccountGroupClassifiedDriverGroups
]:
    config_pth = path.parent.parent / 'config.yml'
    account_source = (
        path.parent.parent.parent / 'tests' / 'test_data' / 'reduced_acc_data.parquet'
    )
    driver_source = (
        path.parent.parent.parent
        / 'tests'
        / 'test_data'
        / 'reduced_driver_data.parquet'
    )

    data_loading_method = DataLoadingMethodEnum.POLARS

    lh_params = parse_yaml(config_pth)
    lh_params.load_data_params.selected_method = data_loading_method
    lh_params.accounts = [
        dts.AccountType(acc)
        for acc in ['volume', 'net_revenue', 'cogs_total', 't_w_total', 'gross_margin']
    ]
    lh_params.segment = dts.ProductType('Residential')
    lh_params.region = dts.LocationType('North America')

    data = load_data_driver_ranking(
        lh_params=lh_params, account_source=account_source, driver_source=driver_source
    )

    return lh_params, dts.AccountGroupClassifiedDriverGroups(
        accounts=data.accounts,
        classified_drivers=data.classified_drivers,
        np_dtype=data.np_dtype,
    )


def perform_driver_analysis(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    general_params: GeneralParams,
    da_params: DriverAnalysisParams,
    output_params: OutputParams,
) -> dts.AccountGroupSelectedDrivers:
    da_info = analyze_drivers_full(
        accounts_drivers_info=DriverAnalysisInput(
            accounts=accounts_drivers_info.accounts,
            classified_drivers=accounts_drivers_info.classified_drivers,
            np_dtype=accounts_drivers_info.np_dtype,
        ),
        general_params=general_params,
        da_params=da_params,
        output_params=output_params,
    )

    return accounts_drivers_info.select_drivers(da_info.selected_drivers)

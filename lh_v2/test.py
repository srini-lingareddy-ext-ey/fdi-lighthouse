import pathlib as pth

import lh_v2.datatypes as dts
from lh_v2.driver_analysis import analyze_drivers_full
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

path = pth.Path.cwd()

acc_pth = path.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
driv_pth = path.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

config_pth = path / 'config.yml'

params = parse_yaml(config_pth)

dr_data = load_data_driver_ranking(
    lh_params=params, account_source=acc_pth, driver_source=driv_pth
)

analysis_output = analyze_drivers_full(
    accounts_drivers_info=dr_data,
    general_params=params.general_params,
    da_params=params.driver_analysis_params,
)


def print_test_ranking_dict(
    test_ranking_dict: dict[
        dts.AccountType,
        dict[dts.DriverClassification, dict[str, dict[dts.DriverName, float]]],
    ],
    best_lags: dict[
        dts.AccountType, dict[dts.DriverClassification, dict[dts.DriverName, int]]
    ],
    account: dts.AccountType,
    class_: dts.DriverClassification,
):
    acc_results = test_ranking_dict[account]

    print(f'Driver Classification - {class_}\n')
    for method in acc_results[class_].keys():
        print(f'\tRanking Method - {method}\n')
        for driver in acc_results[class_][method].keys():
            print(f'\t\tDriver - {driver} | Lag - {best_lags[account][class_][driver]}')
            print(f'\t\t\tValue - {acc_results[class_][method][driver]}\n')
    return

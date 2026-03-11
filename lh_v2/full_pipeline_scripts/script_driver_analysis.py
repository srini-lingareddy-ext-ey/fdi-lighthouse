import pathlib as pth
import pprint

import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
from lh_v2.driver_analysis import driver_ranking_debug
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

b_print_debug = True

path = pth.Path.cwd()

acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

config_pth = path.parent / 'config.yml'

params = parse_yaml(config_pth)

dr_data = load_data_driver_ranking(
    lh_params=params, account_source=acc_pth, driver_source=driv_pth
)

analysis_results, analysis_metrics, best_lags = driver_ranking_debug(
    accounts_drivers_info=dr_data, da_params=params.driver_analysis_params
)

if b_print_debug:
    for account in dr_data.accounts.get_ordered_accounts():
        for classification in dr_data.classified_drivers.get_ordered_classifications():
            print(f'Account: {account} | Classification: {classification}')
            pprint.pprint(analysis_results[account][classification], sort_dicts=False)
            print()

    for account in dr_data.accounts.get_ordered_accounts():
        for classification in dr_data.classified_drivers.get_ordered_classifications():
            print(f'Account: {account} | Classification: {classification}')
            pprint.pprint(analysis_metrics[account][classification], sort_dicts=False)
            print()

for account in dr_data.accounts.get_ordered_accounts():
    for classification in dr_data.classified_drivers.get_ordered_classifications():
        print(f'Account: {account} | Classification: {classification}')
        for driver in analysis_metrics[account][classification].keys():
            rank = int(
                analysis_metrics[account][classification][driver][
                    rdt.DriverRankingMetric.FINAL_RANK
                ]
            )
            print(f'\tDriver: {driver} | Rank: {rank}')
        print()

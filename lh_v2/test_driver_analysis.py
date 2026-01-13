# pyright: reportUnusedImport=false

import pathlib as pth
import pprint

from lh_v2.driver_analysis import analyze_drivers_full, driver_ranking_debug
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

analysis_results, best_lags = driver_ranking_debug(
    accounts_drivers_info=dr_data, da_params=params.driver_analysis_params
)

pprint.pprint(
    analysis_results[dr_data.accounts.get_ordered_accounts()[0]][
        dr_data.classified_drivers.get_ordered_classifications()[0]
    ],
    sort_dicts=False,
)
pprint.pprint(
    best_lags[dr_data.accounts.get_ordered_accounts()[0]][
        dr_data.classified_drivers.get_ordered_classifications()[0]
    ],
    sort_dicts=False,
)

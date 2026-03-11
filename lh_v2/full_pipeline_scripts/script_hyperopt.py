import pathlib as pth

import lh_v2.datatypes as dts
from lh_v2.forecasting import train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

path = pth.Path.cwd()

acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

config_pth = path.parent / 'config.yml'

params = parse_yaml(config_pth)

params.accounts = [dts.AccountType('volume')]

dr_data = load_data_driver_ranking(
    lh_params=params, account_source=acc_pth, driver_source=driv_pth
)

selected_drivers_ = {
    'External': [
        'PPI for HVAC',
        'Construction Spend - Commercial',
        'Interest Rates: 90 Day',
    ],
    'Internal': ['Satisfaction Score: Residential Heating Systems'],
}

selected_drivers: dict[
    dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
] = {}
selected_drivers[dts.AccountType('volume')] = {}
for class_ in selected_drivers_.keys():
    selected_drivers[dts.AccountType('volume')][dts.DriverClassification(class_)] = []
    for driver in selected_drivers_[class_]:
        selected_drivers[dts.AccountType('volume')][
            dts.DriverClassification(class_)
        ].append(dts.DriverName(driver))

classifications: dict[
    dts.AccountType, dict[dts.DriverName, dts.DriverClassification]
] = {}
for account in selected_drivers.keys():
    classifications[account] = {}
    for class_ in selected_drivers[account].keys():
        for driver in selected_drivers[account][class_]:
            classifications[account][driver] = class_


lags: dict[dts.AccountType, dict[dts.DriverName, int]] = {}
for account in selected_drivers.keys():
    lags[account] = {}
    for class_ in selected_drivers[account]:
        for driver in selected_drivers[account][class_]:
            lags[account][driver] = 0


training_input = ModelTrainingInput(
    accounts_drivers=dr_data.select_drivers(selected_drivers=selected_drivers),
    lags=lags,
    classifications=classifications,
)

training_results = train_and_validate_models(
    model_training_info=training_input,
    general_params=params.general_params,
    af_params=params.account_forecast_params,
    df_params=params.driver_forecast_params,
    output_params=params.output_params,
)

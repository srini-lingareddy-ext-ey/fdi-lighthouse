import datetime
import pathlib as pth
import time

import numpy as np
from dateutil.relativedelta import relativedelta

import lh_v2.datatypes as dts
from lh_v2.driver_analysis import analyze_drivers_full
from lh_v2.driver_analysis.driver_analysis_types import DriverAnalysisInput
from lh_v2.forecasting import create_account_forecasts, train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingInput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml
from lh_v2.shared import ArrayF

b_time_parts = True
b_plot = False


def prepend_data(
    accounts_drivers_info: dts.AccountGroupClassifiedDriverGroups,
    n_repeats: int,
) -> dts.AccountGroupClassifiedDriverGroups:
    n_data = accounts_drivers_info.accounts.arr.shape[1]
    arr_new_accounts: ArrayF = np.zeros(
        (
            accounts_drivers_info.accounts.arr.shape[0],
            n_data * n_repeats,
        ),
        dtype=accounts_drivers_info.np_dtype,
    )
    arr_new_drivers: ArrayF = np.zeros(
        (
            accounts_drivers_info.classified_drivers.arr.shape[0],
            n_data * n_repeats,
        ),
        dtype=accounts_drivers_info.np_dtype,
    )
    for i in range(n_repeats):
        arr_new_accounts[
            :,
            i * n_data : (i + 1) * n_data,
        ] = accounts_drivers_info.accounts.arr
        arr_new_drivers[
            :,
            i * n_data : (i + 1) * n_data,
        ] = accounts_drivers_info.classified_drivers.arr

    dates_new: dict[datetime.date, int] = {}
    start_date = min(accounts_drivers_info.accounts.dates[0].keys()) - relativedelta(
        months=n_data * (n_repeats - 1)
    )
    print(f'New start date after prepending data: {start_date}')
    for i in range(n_data * n_repeats):
        dates_new[start_date + relativedelta(months=i)] = i

    dates_acc: list[dict[datetime.date, int]] = []
    for _ in range(len(accounts_drivers_info.accounts)):
        dates_acc.append(dates_new.copy())

    dates_drv: list[dict[datetime.date, int]] = []
    for _ in range(accounts_drivers_info.classified_drivers.arr.shape[0]):
        dates_drv.append(dates_new.copy())

    return dts.AccountGroupClassifiedDriverGroups(
        accounts=dts.AccountGroupInfo(
            arr=arr_new_accounts,
            account_map=accounts_drivers_info.accounts.account_map,
            dates=dates_acc,
            segment_type=accounts_drivers_info.accounts.segment_type,
            region_type=accounts_drivers_info.accounts.region_type,
            np_dtype=accounts_drivers_info.np_dtype,
        ),
        classified_drivers=dts.ClassifiedDriverGroups(
            arr=arr_new_drivers,
            classification_groups=accounts_drivers_info.classified_drivers.classification_groups,
            maps=accounts_drivers_info.classified_drivers.maps,
            dates=dates_drv,
            np_dtype=accounts_drivers_info.np_dtype,
        ),
        np_dtype=accounts_drivers_info.np_dtype,
    )


if __name__ == '__main__':
    start_time = time.time()
    path = pth.Path.cwd()

    acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
    driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

    config_pth = path.parent / 'config.yml'

    last_time = time.time()
    # Read configuration parameters
    lh_params = parse_yaml(config_pth)

    # Load account and driver data
    dr_data = load_data_driver_ranking(
        lh_params=lh_params, account_source=acc_pth, driver_source=driv_pth
    )

    # Prepend data to increase size for testing
    n_repeats = 40  # Adjust this number to increase data size
    lh_params.general_params.training_start_date -= relativedelta(
        months=dr_data.accounts.arr.shape[1] * (n_repeats - 1)
    )
    dr_data = prepend_data(accounts_drivers_info=dr_data, n_repeats=n_repeats)
    dr_data = DriverAnalysisInput(
        accounts=dr_data.accounts,
        classified_drivers=dr_data.classified_drivers,
        np_dtype=dr_data.np_dtype,
    )

    lh_params.general_params.training_end_date -= relativedelta(years=n_repeats)
    lh_params.general_params.validation_start_date -= relativedelta(years=n_repeats)

    print(f'New training start date: {lh_params.general_params.training_start_date}')
    print(f'New training end date: {lh_params.general_params.training_end_date}')
    print(
        f'New validation start date: {lh_params.general_params.validation_start_date}'
    )
    print(f'New validation end date: {lh_params.general_params.validation_end_date}')
    print(f'New testing start date: {lh_params.general_params.testing_start_date}')
    print(f'New testing end date: {lh_params.general_params.testing_end_date}')

    load_data_time = time.time() - last_time
    last_time = time.time()

    # Perform full driver analysis to select drivers and optimal lags
    analysis_results = analyze_drivers_full(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
        output_params=lh_params.output_params,
    )

    driver_analysis_time = time.time() - last_time
    last_time = time.time()

    # Prepare model training input
    training_input = ModelTrainingInput(
        accounts_drivers=dr_data.select_drivers(analysis_results.selected_drivers),
        lags=analysis_results.format_lags(),
        classifications=analysis_results.format_classifications(),
    )

    # Train and validate models using the selected drivers and lags
    training_results = train_and_validate_models(
        model_training_info=training_input,
        general_params=lh_params.general_params,
        af_params=lh_params.account_forecast_params,
        df_params=lh_params.driver_forecast_params,
        output_params=lh_params.output_params,
    )

    model_training_time = time.time() - last_time
    last_time = time.time()

    # Prepare model forecasting input
    forecasting_input = ModelForecastingInput(
        accounts_drivers=dr_data.select_drivers(analysis_results.selected_drivers),
        lags=analysis_results.format_lags(),
        classifications=analysis_results.format_classifications(),
        selected_model=training_results.select_forecast_methods(),
        best_params=training_results.format_best_params(),
        validation_errors=training_results.get_selected_methods_errors(
            selected_methods=training_results.select_forecast_methods(),
            actuals=dr_data.accounts,
            val_date_range=(
                lh_params.general_params.validation_start_date,
                lh_params.general_params.validation_end_date,
            ),
        ),
    )

    # Create final account forecasts
    forecasting_results = create_account_forecasts(
        forecasting_input=forecasting_input,
        general_params=lh_params.general_params,
        df_params=lh_params.driver_forecast_params,
        output_params=lh_params.output_params,
    )

    model_forecasting_time = time.time() - last_time
    last_time = time.time()
    final_time = time.time()

    if b_time_parts:
        print('-' * 80)
        print('Time Summary:')
        print(f'Loading Data: {load_data_time:.6f} seconds.')
        print(f'Driver Analysis: {driver_analysis_time:.6f} seconds.')
        print(f'Model Training and Validation: {model_training_time:.6f} seconds.')
        print(f'Model Forecasting: {model_forecasting_time:.6f} seconds.')
        print('-' * 80)
    print('-' * 80)
    print(f'Total workflow took {final_time - start_time:.6f} seconds.')
    avg_time_per_account = (final_time - start_time) / len(lh_params.accounts)
    print(f'Average time per account: {avg_time_per_account:.6f} seconds.')
    print('-' * 80)

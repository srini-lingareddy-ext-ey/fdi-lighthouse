import pathlib as pth
import time
from pprint import pprint
from typing import Any

import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.driver_analysis import DriverAnalysisOutputLLM, analyze_drivers_llm
from lh_v2.forecasting import create_account_forecasts, train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingInput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

b_print_outputs = False
b_time_parts = True
b_use_sample_config = False


def basic_llm_select_drivers(
    analysis_output: DriverAnalysisOutputLLM,
    n_drivers_per_class: int,
):
    selected_drivers: dict[
        dts.AccountType, dict[dts.DriverClassification, list[dts.DriverName]]
    ] = {}
    for account in analysis_output.selectable_drivers.keys():
        selected_drivers[account] = {}
        for classification in analysis_output.selectable_drivers[account].keys():
            selected_drivers[account][classification] = (
                analysis_output.selectable_drivers[account][classification][
                    :n_drivers_per_class
                ]
            )

    return selected_drivers


def make_sample_config_info() -> dict[str, Any]:
    sample_config = {}
    sample_config['use_default_params'] = True
    sample_config['accounts'] = ['volume']
    sample_config['segment'] = 'Residential'
    sample_config['region'] = 'North America'
    sample_config['general_params'] = {
        'training_start_date': '2019-01-01',
        'training_end_date': '2022-12-01',
        'validation_start_date': '2023-01-01',
        'validation_end_date': '2023-12-01',
        'testing_start_date': '2024-01-01',
        'testing_end_date': '2024-06-01',
    }
    return sample_config


if __name__ == '__main__':
    start_time = time.time()
    path = pth.Path.cwd()

    acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
    driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

    if b_use_sample_config:
        config_pth = make_sample_config_info()
    else:
        config_pth = path.parent / 'config.yml'

    last_time = time.time()
    # Read configuration parameters and load data
    lh_params = parse_yaml(config_pth)

    # Load account and driver data
    dr_data = load_data_driver_ranking(
        lh_params=lh_params, account_source=acc_pth, driver_source=driv_pth
    )
    load_data_time = time.time() - last_time
    last_time = time.time()

    # Perform initial driver analysis to be sent to LLM for driver selection
    analysis_results = analyze_drivers_llm(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
    )
    driver_analysis_time = time.time() - last_time
    last_time = time.time()

    if b_print_outputs:
        print('Selectable Drivers for LLM Selection:\n')
        pprint(analysis_results.selectable_drivers, sort_dicts=False)
        print('-' * 50)
        print('Driver Ranking Metrics from Statistical Analysis:\n')
        pprint(analysis_results.metrics, sort_dicts=False)
        print('-' * 50)

    selected_drivers = basic_llm_select_drivers(
        analysis_output=analysis_results,
        n_drivers_per_class=lh_params.driver_analysis_params.n_final_drivers_per_classification,
    )

    # Prepare model training input
    training_input = ModelTrainingInput(
        accounts_drivers=dr_data.select_drivers(selected_drivers),
        lags=analysis_results.format_lags(selected_drivers=selected_drivers),
        classifications=analysis_results.format_classifications(
            selected_drivers=selected_drivers
        ),
    )

    # Prepare model forecasting input
    training_results = train_and_validate_models(
        model_training_info=training_input,
        general_params=lh_params.general_params,
        af_params=lh_params.account_forecast_params,
        df_params=lh_params.driver_forecast_params,
    )
    model_training_time = time.time() - last_time
    last_time = time.time()

    if b_print_outputs:
        print('Model Validation Metrics for LLM Model Selection:\n')
        pprint(
            training_results.get_llm_forecast_metrics(
                [metric for metric in aft.AccountValidationMetricEnum]
            ),
            sort_dicts=False,
        )
        print('-' * 50)

    # Prepare model forecasting input
    forecasting_input = ModelForecastingInput(
        accounts_drivers=dr_data.select_drivers(selected_drivers),
        lags=analysis_results.format_lags(selected_drivers=selected_drivers),
        classifications=analysis_results.format_classifications(
            selected_drivers=selected_drivers
        ),
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
    print(
        f'Full single instance LLM workflow took {final_time - start_time:.6f} seconds.'
    )
    per_acc_time = (final_time - start_time) / len(
        forecasting_results.accounts_forecasts
    )
    print(f'Per account avg time: {per_acc_time:.6f} seconds.')
    print('-' * 80)

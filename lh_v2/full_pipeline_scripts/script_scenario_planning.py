import pathlib as pth
import time

import lh_v2.datatypes as dts
from lh_v2.account_reconciliation import apply_account_reconciliation
from lh_v2.data_analysis.account_plotting_util import plot_account_forecasts
from lh_v2.datatypes.scenario_planning_types import SPDriverScenarioEnum, SPExtremaCase
from lh_v2.driver_analysis import analyze_drivers_full
from lh_v2.forecasting import create_account_forecasts, train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingInput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml
from lh_v2.scenario_planning.scenario_creation import create_scenarios
from lh_v2.scenario_planning.scenario_planning_types import ScenarioPlanningInput

b_time_parts = True
b_save_csv = False
b_plot = True
scenario_case: SPDriverScenarioEnum = SPDriverScenarioEnum.HIGH

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

    selected_info = dr_data.select_drivers(
        selected_drivers=analysis_results.selected_drivers
    )

    # Prepare model forecasting input
    forecasting_input = ModelForecastingInput(
        accounts_drivers=selected_info,
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

    reconciliation_results = apply_account_reconciliation(
        forecasting_data=forecasting_results,
        reconciliation_params=lh_params.account_reconciliation_params,
        output_params=lh_params.output_params,
    )

    account_reconciliation_time = time.time() - last_time
    last_time = time.time()

    scenario_input = ScenarioPlanningInput(
        accounts_drivers=dts.AccountGroupSelectedDrivers(
            accounts=reconciliation_results.accounts_forecasts,
            drivers=selected_info.drivers,
            np_dtype=selected_info.np_dtype,
        ),
        lags=analysis_results.format_lags(),
        classifications=analysis_results.format_classifications(),
        selected_model=training_results.select_forecast_methods(),
        best_params=training_results.format_best_params(),
    )

    all_drivers = set()
    for account in scenario_input.accounts_drivers.accounts.get_ordered_accounts():
        all_drivers.update(
            scenario_input.accounts_drivers[account].drivers.get_ordered_drivers()
        )

    scenario = {driver: scenario_case for driver in all_drivers}

    scenario_results = create_scenarios(
        scenario_input=scenario_input,
        lh_params=lh_params,
        scenario=scenario,
    )
    scenario_planning_time = time.time() - last_time
    last_time = time.time()

    final_time = time.time()

    if b_time_parts:
        print('-' * 80)
        print('Time Summary:')
        print(f'Loading Data: {load_data_time:.6f} seconds.')
        print(f'Driver Analysis: {driver_analysis_time:.6f} seconds.')
        print(f'Model Training and Validation: {model_training_time:.6f} seconds.')
        print(f'Model Forecasting: {model_forecasting_time:.6f} seconds.')
        print(f'Account Reconciliation: {account_reconciliation_time:.6f} seconds.')
        print(f'Scenario Planning: {scenario_planning_time:.6f} seconds.')
        print('-' * 80)
    print('-' * 80)
    print(f'Total workflow took {final_time - start_time:.6f} seconds.')
    avg_time_per_account = (final_time - start_time) / len(lh_params.accounts)
    print(f'Average time per account: {avg_time_per_account:.6f} seconds.')
    print('-' * 80)

    if b_plot:
        # Plot scenario planning results
        for account in scenario_results.perturbed_accounts.get_ordered_accounts():
            print(f'Plotting scenarios for {account}')
            plot_account_forecasts(
                accounts={
                    'Base Forecast': reconciliation_results.accounts_forecasts[account],
                    'Perturbed Forecast': scenario_results.perturbed_accounts[account],
                    'Max Scenario Forecast': scenario_results.extrema_accounts[
                        SPExtremaCase.MAX
                    ][account],
                    'Min Scenario Forecast': scenario_results.extrema_accounts[
                        SPExtremaCase.MIN
                    ][account],
                },
                forecast_daterange=lh_params.general_params.get_testing_daterange(),
                historicals=dr_data.accounts[account],
                b_plot_historicals=False,
            )

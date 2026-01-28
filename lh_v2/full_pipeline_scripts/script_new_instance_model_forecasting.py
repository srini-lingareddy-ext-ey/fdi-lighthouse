import pathlib as pth
import time

from lh_v2.new_instance_llm_funcs import (
    forecast_models_new_llm,
    select_drivers_llm_new,
    train_models_new_llm,
)

b_time_parts = True

if __name__ == '__main__':
    start_time = time.time()
    path = pth.Path.cwd()

    config_pth = path.parent / 'config.yml'

    acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
    driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

    start_time = time.time()
    last_time = time.time()
    select_drivers_out = select_drivers_llm_new(
        config_info=config_pth,
        account_data_pth=acc_pth,
        driver_data_pth=driv_pth,
    )
    select_drivers_time = time.time() - last_time
    last_time = time.time()

    training_out = train_models_new_llm(
        config_info=config_pth,
        account_data_pth=acc_pth,
        driver_data_pth=driv_pth,
        selected_drivers=None,
        drivers_lags=None,
    )
    model_training_time = time.time() - last_time
    last_time = time.time()

    forecasting_out = forecast_models_new_llm(
        config_info=config_pth,
        account_data_pth=acc_pth,
        driver_data_pth=driv_pth,
        selected_drivers=None,
        drivers_lags=None,
        selected_models=None,
        model_params=None,
    )
    model_forecasting_time = time.time() - last_time
    last_time = time.time()
    final_time = time.time()

    if b_time_parts:
        print('-' * 80)
        print('Time Summary:')
        print(f'Selecting Drivers: {select_drivers_time:.6f} seconds.')
        print(f'Model Training and Validation: {model_training_time:.6f} seconds.')
        print(f'Model Forecasting: {model_forecasting_time:.6f} seconds.')
        print('-' * 80)
    print('-' * 50)
    print(
        f'Full multi instance LLM workflow took {final_time - start_time:.6f} seconds.'
    )
    per_acc_time = (final_time - start_time) / len(forecasting_out.accounts_forecasts)
    print(f'Per account avg time: {per_acc_time:.6f} seconds.')
    print('-' * 50)

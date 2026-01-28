import pathlib as pth

# import lh_v2.datatypes as dts
import lh_v2.datatypes.forecasting_types.account_forecasting_types as aft
from lh_v2.driver_analysis import analyze_drivers_full
from lh_v2.forecasting import train_and_validate_models
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

if __name__ == '__main__':
    path = pth.Path.cwd()

    acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
    driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

    config_pth = path.parent / 'config.yml'

    # Read configuration parameters
    params = parse_yaml(config_pth)

    # Load account and driver data
    dr_data = load_data_driver_ranking(
        lh_params=params, account_source=acc_pth, driver_source=driv_pth
    )

    # Perform full driver analysis to select drivers and optimal lags
    analysis_results = analyze_drivers_full(
        accounts_drivers_info=dr_data,
        general_params=params.general_params,
        da_params=params.driver_analysis_params,
    )

    # Prepare model training input
    training_input = ModelTrainingInput(
        accounts_drivers=dr_data.select_drivers(analysis_results.selected_drivers),
        lags=analysis_results.format_lags(),
        classifications=analysis_results.format_classifications(),
    )

    # Train and validate models using the selected drivers and lags
    training_results = train_and_validate_models(
        model_training_info=training_input,
        general_params=params.general_params,
        af_params=params.account_forecast_params,
        df_params=params.driver_forecast_params,
    )

    # plot forecast for last account
    training_results.plot_forecast(
        account=dr_data.accounts.get_ordered_accounts()[-1],
        method=aft.AccountForecastingMethodEnum.XGBOOST,
        forecast_daterange=(
            params.general_params.validation_start_date,
            params.general_params.validation_end_date,
        ),
    )

    # Select the best method for each account based on validation metrics (default: RMSE%)
    # selected_models = training_results.select_forecast_methods()

    # # Plot forecast for all accounts using their best method
    # for account in dr_data.accounts.get_ordered_accounts():
    #     best_method = selected_models[account]
    #     print(f"Best method for {account}: {best_method.value}")

    #     training_results.plot_forecast(
    #         account=account,
    #         method=best_method,
    #         forecast_daterange=(
    #             params.general_params.validation_start_date,
    #             params.general_params.validation_end_date,
    #         ),
    #     )

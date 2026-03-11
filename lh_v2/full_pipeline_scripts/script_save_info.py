import pathlib as pth

from lh_v2.account_reconciliation import apply_account_reconciliation
from lh_v2.driver_analysis import analyze_drivers_full
from lh_v2.forecasting.account_forecasting.model_forecasting.forecast_accounts import (
    create_account_forecasts,
)
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingInput,
)
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingInput,
)
from lh_v2.forecasting.account_forecasting.model_validation.train_models import (
    train_and_validate_models,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

path = pth.Path.cwd()

acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

config_pth = path.parent / 'config.yml'


if __name__ == '__main__':
    lh_params = parse_yaml(config_pth)

    dr_data = load_data_driver_ranking(
        lh_params=lh_params, account_source=acc_pth, driver_source=driv_pth
    )

    analysis_results = analyze_drivers_full(
        accounts_drivers_info=dr_data,
        general_params=lh_params.general_params,
        da_params=lh_params.driver_analysis_params,
        output_params=lh_params.output_params,
    )

    training_input = ModelTrainingInput(
        accounts_drivers=dr_data.select_drivers(analysis_results.selected_drivers),
        lags=analysis_results.format_lags(),
        classifications=analysis_results.format_classifications(),
    )

    training_results = train_and_validate_models(
        model_training_info=training_input,
        general_params=lh_params.general_params,
        af_params=lh_params.account_forecast_params,
        df_params=lh_params.driver_forecast_params,
        output_params=lh_params.output_params,
    )

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

    forecasting_results = create_account_forecasts(
        forecasting_input=forecasting_input,
        general_params=lh_params.general_params,
        df_params=lh_params.driver_forecast_params,
        output_params=lh_params.output_params,
    )

    reconciliation_results = apply_account_reconciliation(
        forecasting_data=forecasting_results,
        reconciliation_params=lh_params.account_reconciliation_params,
        output_params=lh_params.output_params,
    )

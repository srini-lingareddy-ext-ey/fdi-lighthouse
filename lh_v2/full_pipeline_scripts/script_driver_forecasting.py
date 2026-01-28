import pathlib as pth

import lh_v2.datatypes as dts
from lh_v2.forecasting.driver_forecasting import (
    DriverForecastingInput,
    create_driver_forecasts,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

if __name__ == '__main__':
    b_plot: bool = True

    path = pth.Path.cwd()

    acc_pth = path.parent.parent / 'sample_data' / 'fact_profitability_1205_v23.csv'
    driv_pth = path.parent.parent / 'sample_data' / 'WIP_Drivers_v16.csv'

    config_pth = path.parent / 'config.yml'

    params = parse_yaml(config_pth)

    dr_data = load_data_driver_ranking(
        lh_params=params,
        account_source=acc_pth,
        driver_source=driv_pth,
    )

    driver_data = dr_data.classified_drivers[dts.DriverClassification('External')]

    lags: dict[dts.DriverName, int] = {
        driver: 9 for driver in driver_data.get_ordered_drivers()
    }

    forecasting_input = DriverForecastingInput(
        drivers=driver_data,
        lags=lags,
        training_daterange=(
            params.general_params.training_start_date,
            params.general_params.training_end_date,
        ),
        forecast_daterange=(
            params.general_params.validation_start_date,
            params.general_params.validation_end_date,
        ),
    )

    forecasting_output = create_driver_forecasts(
        forecasting_info=forecasting_input,
        df_params=params.driver_forecast_params,
    )

    if b_plot:
        plotted_driver = driver_data.get_ordered_drivers()[0]
        forecasting_output.plot_driver_forecasts(
            driver_name=plotted_driver,
            lag=lags[plotted_driver],
            forecast_daterange=(
                params.general_params.validation_start_date,
                params.general_params.validation_end_date,
            ),
        )

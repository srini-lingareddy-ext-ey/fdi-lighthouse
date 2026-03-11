import pathlib as pth

import numpy as np
from dateutil.relativedelta import relativedelta

import lh_v2.datatypes as dts
from lh_v2.data_analysis.driver_plotting_util import plot_driver_forecast
from lh_v2.forecasting.driver_forecasting import (
    DriverForecastingInput,
    create_driver_forecasts,
)
from lh_v2.io.data_loading import load_data_driver_ranking
from lh_v2.params import parse_yaml

b_plot: bool = True

if __name__ == '__main__':
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
        driver: int(np.random.randint(1, 10))
        for driver in driver_data.get_ordered_drivers()
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

    forecasted_drivers = dts.DriverGroup(
        arr=forecasting_output.arr,
        map=forecasting_output.map,
        dates=forecasting_output.dates,
        np_dtype=forecasting_output.np_dtype,
    )

    if b_plot:
        for driver in forecasting_output.get_ordered_drivers():
            plot_driver_forecast(
                driver=forecasted_drivers.get_driver(driver),
                forecast_daterange=(
                    params.general_params.validation_start_date,
                    params.general_params.validation_end_date
                    - relativedelta(months=lags[driver]),
                ),
            )

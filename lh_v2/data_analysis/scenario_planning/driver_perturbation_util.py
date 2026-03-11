import datetime

import matplotlib.pyplot as plt
import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
)
from lh_v2.util import month_dif


def plot_driver_perturbations(
    base_forecasts: dts.DriverGroup,
    perturbed_drivers: dict[SPDriverScenarioEnum, dts.DriverGroup],
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    lags: dict[dts.DriverName, int],
):
    xaxis = np.arange(base_forecasts.arr.shape[1])
    n_training = month_dif(*training_daterange) + 1
    n_forecast = month_dif(*forecast_daterange) + 1

    for driver in base_forecasts.get_ordered_drivers():
        driver_idx = base_forecasts.map[driver]

        if n_forecast > lags[driver]:
            plt.figure()
            plt.plot(
                xaxis[: n_training + lags[driver]],
                base_forecasts.arr[driver_idx, : n_training + lags[driver]],
                label='Historicals',
            )
            for scenario, pert_drivers in perturbed_drivers.items():
                plt.plot(
                    xaxis[n_training + lags[driver] :],
                    pert_drivers.arr[driver_idx, n_training + lags[driver] :],
                    label=scenario.value,
                )

            plt.plot(
                xaxis[n_training + lags[driver] :],
                base_forecasts.arr[driver_idx, n_training + lags[driver] :],
                label='Base Forecast',
                linestyle='--',
                color='black',
            )

            plt.title(f'Scenarios for {driver}')
            plt.legend()
            plt.show()

    return

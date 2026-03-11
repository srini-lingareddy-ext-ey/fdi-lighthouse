import datetime

import matplotlib.pyplot as plt
import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
    SPExtremaCase,
)
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif


def plot_all_scenarios(
    base_forecast: dts.AccountInfo,
    forecast_daterange: tuple[datetime.date, datetime.date],
    arr_scenarios: ArrayF,
    extrema_cases: dict[SPExtremaCase, dict[dts.DriverName, SPDriverScenarioEnum]]
    | None = None,
    lst_scenarios: list[dict[dts.DriverName, SPDriverScenarioEnum]] | None = None,
    b_plot_historicals: bool = False,
):
    xaxis = np.arange(base_forecast.arr.shape[0])
    n_forecast = month_dif(*forecast_daterange) + 1

    scenario_mean = arr_scenarios.mean(axis=1)
    idx_max = scenario_mean.argmax()
    idx_min = scenario_mean.argmin()

    plt.figure()
    if b_plot_historicals:
        plt.plot(
            xaxis[:-n_forecast], base_forecast.arr[:-n_forecast], label='Historicals'
        )
    plt.plot(xaxis[-n_forecast:], arr_scenarios.T)
    plt.plot(
        xaxis[-n_forecast:],
        base_forecast.arr[-n_forecast:],
        label='Base Forecast',
        color='black',
        linewidth=4,
    )
    plt.plot(
        xaxis[-n_forecast:],
        arr_scenarios[idx_max],
        label='Max Scenario',
        color='green',
        linewidth=3,
    )
    plt.plot(
        xaxis[-n_forecast:],
        arr_scenarios[idx_min],
        label='Min Scenario',
        color='red',
        linewidth=3,
    )
    if extrema_cases is not None and lst_scenarios is not None:
        for case in extrema_cases.keys():
            plt.plot(
                xaxis[-n_forecast:],
                arr_scenarios[lst_scenarios.index(extrema_cases[case])],
                label=case.value,
                linestyle='--',
                color='black',
                linewidth=3,
            )
    plt.title('All Scenarios')
    plt.legend()
    plt.show()
    return

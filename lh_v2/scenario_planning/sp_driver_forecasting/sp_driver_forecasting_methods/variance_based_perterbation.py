import datetime

import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import SPDriverScenarioEnum
from lh_v2.params.scenario_planning_params import (
    BaseSPDriverForecastingMethodParams,
    SPVARBasedDriverForecastingParams,
)
from lh_v2.shared import ArrayF
from lh_v2.stats import detect_seasonality, std_deseasonalized, std_peacewise_slr_single

from .abstract_class import AbstractSPDriverPerturbationMethod


class VarianceBasedDriverPerturbationMethod(AbstractSPDriverPerturbationMethod):
    def __init__(
        self,
        base_driver_forecasts: dts.DriverGroup,
        training_daterange: tuple[datetime.date, datetime.date],
        forecasting_daterange: tuple[datetime.date, datetime.date],
        params: BaseSPDriverForecastingMethodParams,
        best_lags: dict[dts.DriverName, int],
    ):
        self.base_driver_forecasts = base_driver_forecasts
        self.training_daterange = training_daterange
        self.forecasting_daterange = forecasting_daterange
        assert isinstance(params, SPVARBasedDriverForecastingParams)
        self.params = params
        self.best_lags = best_lags
        return

    def _convert_scenario_to_array(
        self,
        scenario: dict[dts.DriverName, SPDriverScenarioEnum],
    ) -> ArrayF:
        arr: ArrayF = np.zeros(
            (len(self.base_driver_forecasts),),
            dtype=self.base_driver_forecasts.np_dtype,
        )
        for idx, driver in enumerate(self.base_driver_forecasts.get_ordered_drivers()):
            match scenario.get(driver, SPDriverScenarioEnum.AVG):
                case SPDriverScenarioEnum.AVG:
                    arr[idx] = 0.0
                case SPDriverScenarioEnum.HIGH:
                    arr[idx] = 1.0
                case SPDriverScenarioEnum.LOW:
                    arr[idx] = -1.0
                case SPDriverScenarioEnum.VERY_HIGH:
                    arr[idx] = 2.0
                case SPDriverScenarioEnum.VERY_LOW:
                    arr[idx] = -2.0
        return arr

    def perturb_forecasts(
        self, scenario: dict[dts.DriverName, SPDriverScenarioEnum]
    ) -> dts.DriverGroup:
        perturbed_drivers = self.base_driver_forecasts.copy()
        arr_perturbation = self._convert_scenario_to_array(scenario=scenario)

        for idx, driver in enumerate(perturbed_drivers.get_ordered_drivers()):
            if self.has_forecast(driver_name=driver):
                n_forecast_vals = self.get_n_forecast_values(driver_name=driver)
                training_data = self.get_training_data(driver_name=driver)
                arr_time = (
                    np.arange(n_forecast_vals, dtype=perturbed_drivers.np_dtype) + 1
                ) * self.params.time_scaling_factor

                # Check seasonality strength for each period
                strongest_seasonality = 0.0
                strongest_period = 0
                for period in self.params.period_checks:
                    seasonal_strength = detect_seasonality(
                        ts=training_data,
                        period=period,
                    )
                    if seasonal_strength > strongest_seasonality:
                        strongest_seasonality = seasonal_strength
                        strongest_period = period

                # Decide whether to deseasonalize based on threshold
                if strongest_seasonality < self.params.seasonality_threshold:
                    # Driver not seasonal, use raw additive σ
                    driver_std = std_peacewise_slr_single(
                        ts=training_data,
                        n_months_per=self.params.multi_slr_seg_len,
                    )
                    perturbed_drivers.arr[idx, -n_forecast_vals:] += (
                        arr_time * driver_std * arr_perturbation[idx]
                    )
                else:
                    # Driver seasonal, use multiplicative (hybrid) approach
                    # CV = deseasonalized σ / |mean|, scaled by time and scenario
                    driver_std = std_deseasonalized(
                        ts=training_data,
                        n_months_per=self.params.multi_slr_seg_len,
                        period=strongest_period,
                    )
                    driver_mean = np.mean(training_data)
                    if driver_mean == 0:
                        cv = 0.0
                    else:
                        cv = driver_std / abs(driver_mean)

                    scale = np.maximum(
                        0.0,
                        1 + arr_time * cv * arr_perturbation[idx],
                    )
                    perturbed_drivers.arr[idx, -n_forecast_vals:] *= scale
        return perturbed_drivers

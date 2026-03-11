import datetime
from abc import ABC, abstractmethod

import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
)
from lh_v2.params.scenario_planning_params import BaseSPDriverForecastingMethodParams
from lh_v2.shared import ArrayF
from lh_v2.util import month_dif


class AbstractSPDriverPerturbationMethod(ABC):
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
        self.params = params
        self.best_lags = best_lags
        return

    def get_training_data(self, driver_name: dts.DriverName) -> ArrayF:
        driver = self.base_driver_forecasts.get_driver(driver_name=driver_name)
        if min(driver.dates.keys()) > self.training_daterange[0]:
            return driver.apply_daterange(end_date=self.training_daterange[1]).arr
        else:
            return driver.apply_daterange(
                start_date=self.training_daterange[0],
                end_date=self.training_daterange[1],
            ).arr

    def get_n_forecast_values(self, driver_name: dts.DriverName) -> int:
        return (
            month_dif(self.forecasting_daterange[0], self.forecasting_daterange[1])
            + 1
            - self.best_lags[driver_name]
        )

    def has_forecast(self, driver_name: dts.DriverName) -> bool:
        return self.get_n_forecast_values(driver_name) > 0

    def get_base_forecasts(self, driver_name: dts.DriverName) -> ArrayF:
        if not self.has_forecast(driver_name):
            return np.array([], dtype=self.base_driver_forecasts.np_dtype)
        driver = self.base_driver_forecasts.get_driver(driver_name=driver_name)
        return driver.apply_daterange(start_date=self.forecasting_daterange[0]).arr

    @abstractmethod
    def perturb_forecasts(
        self, scenario: dict[dts.DriverName, SPDriverScenarioEnum]
    ) -> dts.DriverGroup:
        raise NotImplementedError

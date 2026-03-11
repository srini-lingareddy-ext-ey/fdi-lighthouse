import datetime

import lh_v2.datatypes as dts
from lh_v2.data_analysis.scenario_planning.driver_perturbation_util import (
    plot_driver_perturbations,
)
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverForecastingMethodEnum,
)
from lh_v2.params import SPDriverForecastingParams
from lh_v2.scenario_planning.sp_driver_forecasting import (
    make_all_driver_perturbations,
)


def forecast_driver_perturbations_test(
    info: dts.ClassifiedDriverGroups[dts.DriverClassification] | dts.DriverGroup,
    training_daterange: tuple[datetime.date, datetime.date],
    forecast_daterange: tuple[datetime.date, datetime.date],
    sp_df_params: SPDriverForecastingParams,
    perturbation_method: SPDriverForecastingMethodEnum,
    driver_classifications: list[dts.DriverClassification] | None = None,
    lags: dict[dts.DriverName, int] | None = None,
):
    if isinstance(info, dts.ClassifiedDriverGroups):
        if driver_classifications is None or len(driver_classifications) == 0:
            info = info.combine()
        else:
            info_classes: list[dts.DriverGroup] = []
            for class_ in driver_classifications:
                info_classes.append(info[class_])
            info = info_classes[0]
            for dg_class in info_classes[1:]:
                info += dg_class

    if lags is None:
        lags = {driver: 0 for driver in info.get_ordered_drivers()}

    sp_df_params.selected_perturbation_method = perturbation_method

    driver_scenarios = make_all_driver_perturbations(
        driver_base_forecast=info,
        training_daterange=training_daterange,
        forecast_daterange=forecast_daterange,
        lags=lags,
        sp_df_params=sp_df_params,
    )

    plot_driver_perturbations(
        base_forecasts=info,
        perturbed_drivers=driver_scenarios,
        training_daterange=training_daterange,
        forecast_daterange=forecast_daterange,
        lags=lags,
    )

    return

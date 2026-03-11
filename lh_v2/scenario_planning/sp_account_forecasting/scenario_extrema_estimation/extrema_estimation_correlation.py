import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
    SPExtremaCase,
)
from lh_v2.params import GeneralParams
from lh_v2.shared import ArrayF
from lh_v2.stats import pearson_correlation


def extrema_estimation_correlation_method(
    account_drivers_base_forecasts: dts.AccountDriverGroup,
    general_params: GeneralParams,
    lags: dict[dts.DriverName, int],
) -> dict[SPExtremaCase, dict[dts.DriverName, SPDriverScenarioEnum]]:
    training_data = account_drivers_base_forecasts.apply_daterange(
        start_date=general_params.training_start_date,
        end_date=general_params.validation_end_date,
        best_lags=lags,
        b_training=True,
    )

    corrs: ArrayF = np.zeros(
        (training_data.drivers.arr.shape[0],), dtype=training_data.np_dtype
    )
    for idx in range(training_data.drivers.arr.shape[0]):
        corrs[idx] = pearson_correlation(
            training_data.drivers.arr[idx, :], training_data.account.arr
        )

    scenarios_estimation: dict[
        SPExtremaCase, dict[dts.DriverName, SPDriverScenarioEnum]
    ] = {
        SPExtremaCase.MAX: {},
        SPExtremaCase.MIN: {},
    }
    for idx, driver in enumerate(training_data.drivers.get_ordered_drivers()):
        if corrs[idx] > 0.0:
            scenarios_estimation[SPExtremaCase.MAX][driver] = (
                SPDriverScenarioEnum.VERY_HIGH
            )
            scenarios_estimation[SPExtremaCase.MIN][driver] = (
                SPDriverScenarioEnum.VERY_LOW
            )
        elif corrs[idx] < 0.0:
            scenarios_estimation[SPExtremaCase.MAX][driver] = (
                SPDriverScenarioEnum.VERY_LOW
            )
            scenarios_estimation[SPExtremaCase.MIN][driver] = (
                SPDriverScenarioEnum.VERY_HIGH
            )
        else:
            scenarios_estimation[SPExtremaCase.MAX][driver] = SPDriverScenarioEnum.AVG
            scenarios_estimation[SPExtremaCase.MIN][driver] = SPDriverScenarioEnum.AVG

    return scenarios_estimation

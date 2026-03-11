import itertools

import numpy as np

import lh_v2.datatypes as dts
from lh_v2.datatypes.scenario_planning_types import (
    SPDriverScenarioEnum,
    SPExtremaCase,
)
from lh_v2.forecasting.account_forecasting.account_forecasting_methods import (
    AbstractAccountForecastingMethod,
)
from lh_v2.params import GeneralParams
from lh_v2.params.scenario_planning_params import (
    SPExactExtremaEstimationMethodParams,
    SPSamplingExtremaEstimationMethodParams,
)

from .extrema_estimation_util import combo_to_scenario_dict, combos_to_scenario_arr


def extrema_estimation_sampling_method(
    accounts_drivers_base_forecasts: dts.AccountDriverGroup,
    general_params: GeneralParams,
    all_scenario_forecasts: dict[SPDriverScenarioEnum, dts.DriverGroup],
    estimation_params: SPSamplingExtremaEstimationMethodParams
    | SPExactExtremaEstimationMethodParams,
    account_forecasting_instance: AbstractAccountForecastingMethod,
    lags: dict[dts.DriverName, int],
) -> dict[SPExtremaCase, dict[dts.DriverName, SPDriverScenarioEnum]]:
    """
    Estimate MAX/MIN extrema scenarios by evaluating a random sample of
    all possible driver-scenario combinations using an OLS linear proxy.

    For *d* drivers each with 5 scenario levels there are 5^d combinations.
    We fit a simple linear model   y ≈ X @ β   on the training data, then
    score each sampled combination by its predicted output.  The combos
    producing the highest and lowest predictions are returned as the MAX
    and MIN extrema respectively.

    When the total number of combinations is ≤ n_samples we evaluate all
    of them exhaustively (no randomness).

    Parameters
    ----------
    training_data : DriverGroup
        Historical driver array, shape (n_drivers, n_time).
    target_variable : AccountInfo
        Historical account array, shape (n_time,).
    n_samples : int
        Number of random combos to evaluate.  Clamped to [200, total_combos].

    Returns
    -------
    dict mapping SPExtremaCase → dict[DriverName, SPDriverScenarioEnum]
    """
    n_drivers = accounts_drivers_base_forecasts.drivers.arr.shape[0]
    n_levels = len(list(SPDriverScenarioEnum))
    total_combos = n_levels**n_drivers

    if isinstance(estimation_params, SPSamplingExtremaEstimationMethodParams):
        n_samples = min(estimation_params.n_samples, total_combos)
    else:
        n_samples = total_combos

    if total_combos <= n_samples:
        # Exhaustive: evaluate every combo
        combos = np.array(
            list(itertools.product(range(n_levels), repeat=n_drivers)),
            dtype=np.int32,
        )
    else:
        # Random sampling (uniform over the discrete space)
        rng = np.random.default_rng(seed=42)
        combos = rng.integers(
            low=0, high=n_levels, size=(n_samples, n_drivers), dtype=np.int32
        )

    all_scenario_forecasts_filtered = {
        scenario_enum: driver_group.apply_daterange_lags(
            start_date=general_params.testing_start_date,
            end_date=general_params.testing_end_date,
            lags=lags,
            b_training=False,
        )
        for scenario_enum, driver_group in all_scenario_forecasts.items()
    }

    arr_drivers_scenarios = combos_to_scenario_arr(
        combos=combos,
        all_scenarios_forecast=all_scenario_forecasts_filtered,
    )

    arr_account_scenarios = account_forecasting_instance.apply_vectorized(
        arr_input=arr_drivers_scenarios
    )

    scores = arr_account_scenarios.mean(axis=1)

    # Best MAX combo = highest score, best MIN = lowest score
    max_idx = int(np.argmax(scores))
    min_idx = int(np.argmin(scores))

    return {
        SPExtremaCase.MAX: combo_to_scenario_dict(
            combo=combos[max_idx],
            drivers_ordered=accounts_drivers_base_forecasts.drivers.get_ordered_drivers(),
        ),
        SPExtremaCase.MIN: combo_to_scenario_dict(
            combo=combos[min_idx],
            drivers_ordered=accounts_drivers_base_forecasts.drivers.get_ordered_drivers(),
        ),
    }

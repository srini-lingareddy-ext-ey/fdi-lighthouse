import pytest

from lh_v2.datatypes import (
    AccountGroupClassifiedDriverGroups,
    DriverClassification,
)
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingEnum
from lh_v2.driver_analysis.lagging import lag_handling
from lh_v2.params import DriverAnalysisParams
from lh_v2.params.driver_analysis_params.ranking_params.extra_params import (
    RankingMethodsParams,
)


@pytest.mark.unit
@pytest.mark.driver_analysis
class TestLagHandling:
    def test_select_best_lags_rejects_negative_max_lag(
        self,
        single_driver_accounts_info: AccountGroupClassifiedDriverGroups,
    ) -> None:
        params = DriverAnalysisParams()
        params.lag_params.b_lag = True
        params.lag_params.n_max_lag = -1
        params.lag_params.n_top_considered = 1
        with pytest.raises(AssertionError):
            lag_handling.select_best_lags(
                info=single_driver_accounts_info,
                da_params=params,
            )

    def test_select_best_lags_rejects_too_short_series(
        self,
        single_driver_accounts_info: AccountGroupClassifiedDriverGroups,
    ) -> None:
        params = DriverAnalysisParams()
        params.lag_params.b_lag = True
        params.lag_params.n_max_lag = 6
        params.lag_params.n_top_considered = 1
        with pytest.raises(AssertionError):
            lag_handling.select_best_lags(
                info=single_driver_accounts_info,
                da_params=params,
            )

    def test_select_best_lags_returns_zero_when_lagging_disabled(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.lag_params.b_lag = False
        params.lag_params.n_max_lag = 0

        out, _ = lag_handling.select_best_lags(
            info=sample_accounts_drivers_info,
            da_params=params,
        )

        for account in out:
            for class_ in out[account]:
                for driver in out[account][class_]:
                    assert out[account][class_][driver] == 0

    def test_select_best_lags_enabled_returns_values_within_bounds(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.lag_params.b_lag = True
        params.lag_params.n_max_lag = 1
        params.lag_params.n_top_considered = 1
        params.lag_params.ranking_methods = RankingMethodsParams(
            b_remove=False,
            methods_removed=(),
            b_selected=True,
            methods_selected=(DriverRankingEnum.SPEARMAN_CORRELATION,),
        )
        params.ranking_params.methods = RankingMethodsParams(
            b_remove=False,
            methods_removed=(),
            b_selected=True,
            methods_selected=(DriverRankingEnum.SPEARMAN_CORRELATION,),
        )

        out, _ = lag_handling.select_best_lags(
            info=sample_accounts_drivers_info,
            da_params=params,
        )

        for account in out:
            for class_ in (
                DriverClassification('External'),
                DriverClassification('Internal'),
            ):
                for driver in out[account][class_]:
                    lag = out[account][class_][driver]
                    assert 0 <= lag <= params.lag_params.n_max_lag

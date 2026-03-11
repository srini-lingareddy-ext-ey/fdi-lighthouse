import pytest

from lh_v2.datatypes import AccountGroupClassifiedDriverGroups
from lh_v2.datatypes.driver_analysis_types.ranking_types import (
    DriverRankingEnum,
    DriverRankingMetric,
)
from lh_v2.driver_analysis.driver_ranking import rank
from lh_v2.driver_analysis.driver_ranking.driver_ranking_util import set_lags_to_zero
from lh_v2.params import DriverAnalysisParams
from lh_v2.params.driver_analysis_params.ranking_params.extra_params import (
    RankingMethodsParams,
)


@pytest.mark.unit
@pytest.mark.driver_analysis
class TestRankDrivers:
    def test_rank_returns_expected_metrics_with_real_method(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.lag_params.b_lag = False
        params.lag_params.n_max_lag = 0

        best_lags = set_lags_to_zero(sample_accounts_drivers_info)
        out = rank(
            accounts_drivers_info=sample_accounts_drivers_info,
            ranking_params=params.ranking_params,
            best_lags=best_lags,
            max_lag=0,
        )

        for account in out:
            for class_ in out[account]:
                metrics_by_driver = out[account][class_]
                ranks = []
                for _driver, metrics in metrics_by_driver.items():
                    assert DriverRankingMetric.FINAL_RANK in metrics
                    assert DriverRankingMetric.AVG_RANK in metrics
                    assert isinstance(metrics[DriverRankingMetric.FINAL_RANK], float)
                    assert isinstance(metrics[DriverRankingMetric.AVG_RANK], float)
                    ranks.append(int(metrics[DriverRankingMetric.FINAL_RANK]))

                assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_rank_with_none_best_lags_uses_internal_zero_lags(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.lag_params.b_lag = False
        params.lag_params.n_max_lag = 0

        out = rank(
            accounts_drivers_info=sample_accounts_drivers_info,
            ranking_params=params.ranking_params,
            best_lags=None,
            max_lag=0,
        )
        assert len(out) == len(sample_accounts_drivers_info.accounts.account_map)

    def test_rank_raises_when_selected_method_list_empty(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.ranking_params.methods = RankingMethodsParams(
            b_remove=False,
            methods_removed=(),
            b_selected=True,
            methods_selected=(),
        )
        best_lags = set_lags_to_zero(sample_accounts_drivers_info)

        with pytest.raises(AssertionError):
            rank(
                accounts_drivers_info=sample_accounts_drivers_info,
                ranking_params=params.ranking_params,
                best_lags=best_lags,
                max_lag=0,
            )

    def test_rank_raises_on_conflicting_method_selection_flags(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.ranking_params.methods = RankingMethodsParams(
            b_remove=True,
            methods_removed=(DriverRankingEnum.PCA,),
            b_selected=True,
            methods_selected=(DriverRankingEnum.PEARSON_CORRELATION,),
        )
        best_lags = set_lags_to_zero(sample_accounts_drivers_info)

        with pytest.raises(ValueError):
            rank(
                accounts_drivers_info=sample_accounts_drivers_info,
                ranking_params=params.ranking_params,
                best_lags=best_lags,
                max_lag=0,
            )

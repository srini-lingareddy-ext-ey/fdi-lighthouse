import pytest

from lh_v2.datatypes import (
    AccountGroupClassifiedDriverGroups,
    AccountType,
    DriverClassification,
    DriverName,
)
from lh_v2.datatypes.driver_analysis_types.collinearity_types import (
    CollinearityMethodEnum,
)
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingEnum
from lh_v2.driver_analysis.collinearity import get_collinearity_arr
from lh_v2.driver_analysis.driver_ranking import rank
from lh_v2.params import DriverAnalysisParams
from lh_v2.params.driver_analysis_params.ranking_params.extra_params import (
    RankingMethodsParams,
)


def _flatten_lags(
    zero_best_lags: dict[
        AccountType,
        dict[DriverClassification, dict[DriverName, int]],
    ],
) -> dict[AccountType, dict[DriverName, int]]:
    flattened: dict[AccountType, dict[DriverName, int]] = {}
    for account, class_map in zero_best_lags.items():
        flattened[account] = {}
        for driver_lags in class_map.values():
            flattened[account].update(driver_lags)
    return flattened


@pytest.mark.unit
@pytest.mark.driver_analysis
class TestDriverAnalysisConfigMatrix:
    def test_rank_method_selection_empty_selected_methods_raises(
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
        best_lags: dict[
            AccountType, dict[DriverClassification, dict[DriverName, int]]
        ] = {
            account: {
                classification: {driver: 0 for driver in drivers}
                for classification, drivers in {
                    DriverClassification('External'): (
                        DriverName('ext_1'),
                        DriverName('ext_2'),
                    ),
                    DriverClassification('Internal'): (
                        DriverName('int_1'),
                        DriverName('int_2'),
                    ),
                }.items()
            }
            for account in sample_accounts_drivers_info.accounts.get_ordered_accounts()
        }

        with pytest.raises(AssertionError):
            rank(
                accounts_drivers_info=sample_accounts_drivers_info,
                ranking_params=params.ranking_params,
                best_lags=best_lags,
                max_lag=0,
            )

    def test_remove_collinearity_method_selection_empty_selected_methods_raises(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
        zero_best_lags: dict[
            AccountType, dict[DriverClassification, dict[DriverName, int]]
        ],
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.collinearity_params.b_remove = False
        params.collinearity_params.b_selected = True
        params.collinearity_params.methods_selected = ()
        combined_drivers = sample_accounts_drivers_info.classified_drivers.combine()

        with pytest.raises(AssertionError):
            get_collinearity_arr(
                acc_lst=sample_accounts_drivers_info.accounts.get_ordered_accounts(),
                drivers_combined=combined_drivers,
                da_params=params,
                flattened_lags=_flatten_lags(zero_best_lags),
            )

    @pytest.mark.parametrize(
        ('removed_method', 'selected_method'),
        [
            (DriverRankingEnum.PCA, DriverRankingEnum.PEARSON_CORRELATION),
            (DriverRankingEnum.BORUTA, DriverRankingEnum.SPEARMAN_CORRELATION),
        ],
    )
    def test_rank_conflicting_flags_raise_value_error(
        self,
        removed_method: DriverRankingEnum,
        selected_method: DriverRankingEnum,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.ranking_params.methods = RankingMethodsParams(
            b_remove=True,
            methods_removed=(removed_method,),
            b_selected=True,
            methods_selected=(selected_method,),
        )
        best_lags: dict[
            AccountType, dict[DriverClassification, dict[DriverName, int]]
        ] = {
            account: {
                DriverClassification('External'): {
                    DriverName('ext_1'): 0,
                    DriverName('ext_2'): 0,
                },
                DriverClassification('Internal'): {
                    DriverName('int_1'): 0,
                    DriverName('int_2'): 0,
                },
            }
            for account in sample_accounts_drivers_info.accounts.get_ordered_accounts()
        }
        with pytest.raises(ValueError):
            rank(
                accounts_drivers_info=sample_accounts_drivers_info,
                ranking_params=params.ranking_params,
                best_lags=best_lags,
                max_lag=0,
            )

    def test_remove_collinearity_conflicting_flags_raise_value_error(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
        zero_best_lags: dict[
            AccountType, dict[DriverClassification, dict[DriverName, int]]
        ],
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.collinearity_params.b_remove = True
        params.collinearity_params.methods_removed = (
            CollinearityMethodEnum.CORRELATION,
        )
        params.collinearity_params.b_selected = True
        params.collinearity_params.methods_selected = (CollinearityMethodEnum.VIF,)
        combined_drivers = sample_accounts_drivers_info.classified_drivers.combine()

        with pytest.raises(ValueError):
            get_collinearity_arr(
                acc_lst=sample_accounts_drivers_info.accounts.get_ordered_accounts(),
                drivers_combined=combined_drivers,
                da_params=params,
                flattened_lags=_flatten_lags(zero_best_lags),
            )

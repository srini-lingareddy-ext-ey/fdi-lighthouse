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
from lh_v2.driver_analysis.collinearity import (
    get_collinearity_arr,
    select_final_drivers,
)
from lh_v2.params import DriverAnalysisParams


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


def _empty_preselected_drivers(
    info: AccountGroupClassifiedDriverGroups,
) -> dict[AccountType, dict[DriverClassification, list[DriverName]]]:
    return {
        account: {
            class_: []
            for class_ in info.classified_drivers.get_ordered_classifications()
        }
        for account in info.accounts.get_ordered_accounts()
    }


@pytest.mark.unit
@pytest.mark.driver_analysis
class TestRemoveCollinearity:
    def test_remove_collinearity_applies_rank_plus_collinearity_filter(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
        zero_best_lags: dict[
            AccountType,
            dict[DriverClassification, dict[DriverName, int]],
        ],
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.n_final_drivers_per_classification = 1

        rankings: dict[
            AccountType,
            dict[DriverClassification, dict[DriverName, int]],
        ] = {}
        for account in sample_accounts_drivers_info.accounts.get_ordered_accounts():
            rankings[account] = {
                DriverClassification('External'): {
                    DriverName('ext_1'): 1,
                    DriverName('ext_2'): 2,
                },
                DriverClassification('Internal'): {
                    DriverName('int_1'): 1,
                    DriverName('int_2'): 2,
                },
            }

        combined_drivers = sample_accounts_drivers_info.classified_drivers.combine()
        arr_collinearity_dict = get_collinearity_arr(
            acc_lst=sample_accounts_drivers_info.accounts.get_ordered_accounts(),
            drivers_combined=combined_drivers,
            da_params=params,
            flattened_lags=_flatten_lags(zero_best_lags),
        )
        out, _ = select_final_drivers(
            drivers_map=combined_drivers.map,
            classified_driver_rankings=rankings,
            da_params=params,
            preselected_drivers=_empty_preselected_drivers(
                sample_accounts_drivers_info
            ),
            arr_collinearity_dict=arr_collinearity_dict,
        )

        for account in sample_accounts_drivers_info.accounts.get_ordered_accounts():
            assert out[account][DriverClassification('External')] == [
                DriverName('ext_1')
            ]
            assert out[account][DriverClassification('Internal')] == [
                DriverName('int_2')
            ]

    def test_remove_collinearity_never_exceeds_requested_driver_count(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
        zero_best_lags: dict[
            AccountType,
            dict[DriverClassification, dict[DriverName, int]],
        ],
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.n_final_drivers_per_classification = 2

        rankings: dict[
            AccountType,
            dict[DriverClassification, dict[DriverName, int]],
        ] = {}
        for account in sample_accounts_drivers_info.accounts.get_ordered_accounts():
            rankings[account] = {
                DriverClassification('External'): {
                    DriverName('ext_1'): 1,
                    DriverName('ext_2'): 2,
                },
                DriverClassification('Internal'): {
                    DriverName('int_1'): 1,
                    DriverName('int_2'): 2,
                },
            }

        combined_drivers = sample_accounts_drivers_info.classified_drivers.combine()
        arr_collinearity_dict = get_collinearity_arr(
            acc_lst=sample_accounts_drivers_info.accounts.get_ordered_accounts(),
            drivers_combined=combined_drivers,
            da_params=params,
            flattened_lags=_flatten_lags(zero_best_lags),
        )
        out, _ = select_final_drivers(
            drivers_map=combined_drivers.map,
            classified_driver_rankings=rankings,
            da_params=params,
            preselected_drivers=_empty_preselected_drivers(
                sample_accounts_drivers_info
            ),
            arr_collinearity_dict=arr_collinearity_dict,
        )

        for account in sample_accounts_drivers_info.accounts.get_ordered_accounts():
            external_selected = out[account][DriverClassification('External')]
            internal_selected = out[account][DriverClassification('Internal')]

            assert (
                0 < len(external_selected) <= params.n_final_drivers_per_classification
            )
            assert (
                0 < len(internal_selected) <= params.n_final_drivers_per_classification
            )
            assert len(set(external_selected)) == len(external_selected)
            assert len(set(internal_selected)) == len(internal_selected)

    def test_remove_collinearity_raises_on_conflicting_method_flags(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        minimal_driver_analysis_params: DriverAnalysisParams,
        zero_best_lags: dict[
            AccountType,
            dict[DriverClassification, dict[DriverName, int]],
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

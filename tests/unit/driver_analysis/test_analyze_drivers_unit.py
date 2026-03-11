import pytest

from lh_v2.datatypes import AccountGroupClassifiedDriverGroups
from lh_v2.datatypes.driver_analysis_types.ranking_types import (
    DriverRankingEnum,
    DriverRankingMetric,
)
from lh_v2.driver_analysis import analyze_drivers_full, analyze_drivers_llm
from lh_v2.driver_analysis.driver_analysis_types import (
    DriverAnalysisInput,
    DriverAnalysisOutput,
    DriverAnalysisOutputLLM,
)
from lh_v2.params import DriverAnalysisParams, GeneralParams, OutputParams
from lh_v2.params.driver_analysis_params.ranking_params.extra_params import (
    RankingMethodsParams,
)


@pytest.mark.unit
@pytest.mark.driver_analysis
class TestAnalyzeDriversFull:
    def test_analyze_drivers_full_runs_with_real_dependencies(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        basic_general_params: GeneralParams,
        minimal_driver_analysis_params: DriverAnalysisParams,
        basic_output_params: OutputParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.lag_params.b_lag = False
        params.lag_params.n_max_lag = 0
        params.n_final_drivers_per_classification = 1

        analysis_input = DriverAnalysisInput(
            accounts=sample_accounts_drivers_info.accounts,
            classified_drivers=sample_accounts_drivers_info.classified_drivers,
            np_dtype=sample_accounts_drivers_info.np_dtype,
        )

        out = analyze_drivers_full(
            accounts_drivers_info=analysis_input,
            general_params=basic_general_params,
            da_params=params,
            output_params=basic_output_params,
        )

        assert isinstance(out, DriverAnalysisOutput)

        for account in sample_accounts_drivers_info.accounts.get_ordered_accounts():
            assert account in out.selected_drivers
            assert account in out.metrics
            assert account in out.lags

            for class_ in sample_accounts_drivers_info.classified_drivers.get_ordered_classifications():
                assert class_ in out.selected_drivers[account]
                assert class_ in out.metrics[account]
                assert class_ in out.lags[account]

                selected = out.selected_drivers[account][class_]
                assert len(selected) == params.n_final_drivers_per_classification

                selectable = (
                    sample_accounts_drivers_info.classified_drivers.get_ordered_drivers(
                        class_
                    )
                )
                for driver in selected:
                    assert driver in selectable

                for driver in selectable:
                    assert out.lags[account][class_][driver] == 0
                    assert (
                        DriverRankingMetric.FINAL_RANK
                        in out.metrics[account][class_][driver]
                    )

    def test_analyze_drivers_full_raises_on_invalid_ranking_method_config(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        basic_general_params: GeneralParams,
        minimal_driver_analysis_params: DriverAnalysisParams,
        basic_output_params: OutputParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.ranking_params.methods = RankingMethodsParams(
            b_remove=True,
            methods_removed=(DriverRankingEnum.PCA,),
            b_selected=True,
            methods_selected=(DriverRankingEnum.PEARSON_CORRELATION,),
        )

        analysis_input = DriverAnalysisInput(
            accounts=sample_accounts_drivers_info.accounts,
            classified_drivers=sample_accounts_drivers_info.classified_drivers,
            np_dtype=sample_accounts_drivers_info.np_dtype,
        )

        with pytest.raises(ValueError):
            analyze_drivers_full(
                accounts_drivers_info=analysis_input,
                general_params=basic_general_params,
                da_params=params,
                output_params=basic_output_params,
            )


@pytest.mark.unit
@pytest.mark.driver_analysis
class TestAnalyzeDriversLLM:
    def test_analyze_drivers_llm_returns_selectable_subset_and_metrics(
        self,
        sample_accounts_drivers_info: AccountGroupClassifiedDriverGroups,
        basic_general_params: GeneralParams,
        minimal_driver_analysis_params: DriverAnalysisParams,
    ) -> None:
        params = minimal_driver_analysis_params.model_copy(deep=True)
        params.lag_params.b_lag = False
        params.lag_params.n_max_lag = 0
        params.n_final_drivers_per_classification = 1

        analysis_input = DriverAnalysisInput(
            accounts=sample_accounts_drivers_info.accounts,
            classified_drivers=sample_accounts_drivers_info.classified_drivers,
            np_dtype=sample_accounts_drivers_info.np_dtype,
        )
        out = analyze_drivers_llm(
            accounts_drivers_info=analysis_input,
            general_params=basic_general_params,
            da_params=params,
        )

        assert isinstance(out, DriverAnalysisOutputLLM)
        for account in out.selectable_drivers:
            for class_ in out.selectable_drivers[account]:
                selectable = set(out.selectable_drivers[account][class_])
                metric_keys = set(out.metrics[account][class_].keys())
                assert metric_keys == selectable
                for driver in selectable:
                    assert (
                        DriverRankingMetric.FINAL_RANK
                        in out.metrics[account][class_][driver]
                    )

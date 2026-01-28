"""
Docstring for tests.integration.driver_analysis.test_driver_ranking

These tests verify the driver ranking analysis functionality.
"""

import pytest

from lh_v2.datatypes import AccountGroupClassifiedDriverGroups
from lh_v2.driver_analysis.driver_analysis_types import (
    DriverAnalysisOutput,
)
from lh_v2.params import LighthouseParams


@pytest.mark.integration
@pytest.mark.driver_analysis
class TestDriverAnalysisIntegration:
    """Integration tests for driver analysis."""

    def test_full_driver_analysis_runs(
        self,
        sample_acc_classified_drivers_info: AccountGroupClassifiedDriverGroups,
        sample_params: LighthouseParams,
        sample_analyze_drivers_full: DriverAnalysisOutput,
    ):
        """Test that full driver analysis runs without errors."""

        assert isinstance(sample_analyze_drivers_full, DriverAnalysisOutput)

        for (
            account
        ) in sample_acc_classified_drivers_info.accounts.get_ordered_accounts():
            assert account in sample_analyze_drivers_full.selected_drivers.keys()

            for class_ in sample_acc_classified_drivers_info.classified_drivers.get_ordered_classifications():
                assert (
                    class_
                    in sample_analyze_drivers_full.selected_drivers[account].keys()
                )
                selectable_drivers = sample_acc_classified_drivers_info.classified_drivers.get_ordered_drivers(
                    classification=class_
                )
                selected_drivers = sample_analyze_drivers_full.selected_drivers[
                    account
                ][class_]
                assert (
                    len(selected_drivers)
                    == sample_params.driver_analysis_params.n_final_drivers_per_classification
                )

                for driver in selected_drivers:
                    assert driver in selectable_drivers

                for driver in selectable_drivers:
                    assert (
                        sample_analyze_drivers_full.lags[account][class_][driver] >= 0
                    )
                    assert (
                        sample_analyze_drivers_full.lags[account][class_][driver]
                        <= sample_params.driver_analysis_params.lag_params.n_max_lag
                    )
        return

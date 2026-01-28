"""
Docstring for tests.integration.model_forecasting.test_model_forecasting_integration

These tests verify the model forecasting functionality.
"""

import pytest

from lh_v2.datatypes import AccountGroupClassifiedDriverGroups
from lh_v2.forecasting.account_forecasting.model_forecasting.model_forecasting_types import (
    ModelForecastingOutput,
)
from lh_v2.params import LighthouseParams


@pytest.mark.integration
@pytest.mark.model_forecasting
class TestModelForecastingIntegration:
    """Integration tests for model forecasting."""

    def test_model_forecasting_runs(
        self,
        sample_acc_classified_drivers_info: AccountGroupClassifiedDriverGroups,
        sample_params: LighthouseParams,
        sample_model_forecasting: ModelForecastingOutput,
    ):
        """Test that model forecasting runs without errors."""
        assert isinstance(sample_model_forecasting, ModelForecastingOutput)

        for (
            account
        ) in sample_acc_classified_drivers_info.accounts.get_ordered_accounts():
            assert (
                account
                in sample_model_forecasting.accounts_forecasts.get_ordered_accounts()
            )

        return

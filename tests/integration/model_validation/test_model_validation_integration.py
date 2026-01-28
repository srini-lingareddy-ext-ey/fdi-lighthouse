"""
Docstring for tests.integration.model_validation.test_model_validation_integration

These tests verify the model validation functionality.
"""

import pytest

from lh_v2.datatypes import AccountGroupClassifiedDriverGroups
from lh_v2.forecasting.account_forecasting.model_validation.model_training_types import (
    ModelTrainingOutput,
)
from lh_v2.params import LighthouseParams


@pytest.mark.integration
@pytest.mark.model_validation
class TestModelValidationIntegration:
    """Integration tests for model validation."""

    def test_model_validation_runs(
        self,
        sample_acc_classified_drivers_info: AccountGroupClassifiedDriverGroups,
        sample_params: LighthouseParams,
        sample_model_training_validation: ModelTrainingOutput,
    ):
        """Test that model training and validation runs without errors."""
        assert isinstance(sample_model_training_validation, ModelTrainingOutput)

        for (
            account
        ) in sample_acc_classified_drivers_info.accounts.get_ordered_accounts():
            assert account in sample_model_training_validation.account_map.keys()

        return

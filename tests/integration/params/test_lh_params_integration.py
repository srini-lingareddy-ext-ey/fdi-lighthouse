"""
Integration tests for LighthouseParams functionality.

These tests verify that parameter parsing and validation work correctly.
"""

import pytest

from lh_v2.params import LighthouseParams


@pytest.mark.integration
@pytest.mark.params
class TestLighthouseParamsIntegration:
    """Integration tests for LighthouseParams class."""

    def test_parse_yaml_returns_params(self, parsed_params: LighthouseParams):
        """Test that parsing YAML returns LighthouseParams."""
        assert isinstance(parsed_params, LighthouseParams)
        return

    def test_general_params_configured(self, parsed_params: LighthouseParams):
        """Test that training dates are properly set."""
        assert parsed_params.general_params is not None
        assert parsed_params.general_params.training_start_date is not None
        assert parsed_params.general_params.training_end_date is not None
        assert parsed_params.general_params.validation_start_date is not None
        assert parsed_params.general_params.validation_end_date is not None
        assert parsed_params.general_params.testing_start_date is not None
        assert parsed_params.general_params.testing_end_date is not None

        assert (
            parsed_params.general_params.training_start_date
            < parsed_params.general_params.training_end_date
        )
        assert (
            parsed_params.general_params.validation_start_date
            < parsed_params.general_params.validation_end_date
        )
        assert (
            parsed_params.general_params.testing_start_date
            < parsed_params.general_params.testing_end_date
        )

        assert (
            parsed_params.general_params.training_end_date
            < parsed_params.general_params.validation_start_date
        )
        assert (
            parsed_params.general_params.validation_end_date
            < parsed_params.general_params.testing_start_date
        )
        return

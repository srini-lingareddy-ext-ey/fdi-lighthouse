"""
Unit tests for parameter configuration classes.

These tests verify that parameter parsing and validation work correctly.
"""

import pathlib as pth

import pytest

from lh_v2.params import LighthouseParams, parse_yaml


@pytest.mark.unit
@pytest.mark.params
class TestLighthouseParamsUnit:
    """Unit Tests for LighthouseParams class."""

    def test_default_params_creation(self, default_params: LighthouseParams):
        """Test that default params can be created."""
        assert isinstance(default_params, LighthouseParams)
        assert len(default_params.accounts) > 0
        return


@pytest.mark.unit
@pytest.mark.params
class TestConfigParsingUnit:
    """Tests for configuration file parsing."""

    def test_config_file_exists(self, config_path: pth.Path):
        """Test that config file exists."""
        assert config_path.exists()
        return

    def test_invalid_config_path_raises_error(self):
        """Test that invalid config path raises error."""
        invalid_path = pth.Path('/nonexistent/config.yml')
        with pytest.raises(FileNotFoundError):
            parse_yaml(invalid_path)
        return

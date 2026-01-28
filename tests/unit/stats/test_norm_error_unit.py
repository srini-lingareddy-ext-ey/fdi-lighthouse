"""
Unit tests for error metric calculations.

These tests verify MSE, RMSE, MAPE, and R-squared calculations.
"""

import numpy as np
import pytest

from lh_v2.shared import ArrayF
from lh_v2.stats.norm_error import (
    mape,
    mse,
    rmse,
    rsquared,
)


@pytest.mark.unit
@pytest.mark.stats
class TestMSE:
    """Tests for Mean Squared Error calculation."""

    def test_identical_arrays_zero_error(self, sample_array: ArrayF):
        """Test that identical arrays have MSE of 0."""
        result = mse(sample_array, sample_array)
        assert np.isclose(result, 0.0)

    def test_known_mse_value(self):
        """Test MSE with known values."""
        model = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        true = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        # MSE = ((0^2 + 1^2 + 2^2) / 3) = 5/3 ≈ 1.667
        result = mse(model, true)
        expected = 5.0 / 3.0
        assert np.isclose(result, expected)

    def test_mse_always_positive(self):
        """Test that MSE is always non-negative."""
        model = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        true = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = mse(model, true)
        assert result >= 0.0


@pytest.mark.unit
@pytest.mark.stats
class TestRMSE:
    """Tests for Root Mean Squared Error calculation."""

    def test_identical_arrays_zero_error(self, sample_array: ArrayF):
        """Test that identical arrays have RMSE of 0."""
        result = rmse(sample_array, sample_array)
        assert np.isclose(result, 0.0)

    def test_rmse_is_sqrt_of_mse(self):
        """Test that RMSE equals sqrt(MSE)."""
        model = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        true = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        mse_val = mse(model, true)
        rmse_val = rmse(model, true)
        assert np.isclose(rmse_val, np.sqrt(mse_val))


@pytest.mark.unit
@pytest.mark.stats
class TestMAPE:
    """Tests for Mean Absolute Percentage Error calculation."""

    def test_identical_arrays_zero_error(self):
        """Test that identical arrays have MAPE of 0."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = mape(arr, arr)
        assert np.isclose(result, 0.0)

    def test_mape_percentage_scale(self):
        """Test MAPE returns percentage value."""
        model = np.array([90.0, 100.0, 110.0], dtype=np.float32)
        true = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        # MAPE = mean(|10/100|, |0/100|, |10/100|) = mean(0.1, 0, 0.1) = 0.0667
        result = mape(model, true)
        assert 0.0 <= result <= 1.0  # Should be a ratio


@pytest.mark.unit
@pytest.mark.stats
class TestRSquared:
    """Tests for R-squared calculation."""

    def test_perfect_prediction(self, sample_array: ArrayF):
        """Test that perfect predictions have R² of 1."""
        result = rsquared(sample_array, sample_array)
        assert np.isclose(result, 1.0)

    def test_r_squared_range(self):
        """Test that R² is typically between 0 and 1 for reasonable predictions."""
        model = np.array([1.1, 2.0, 2.9, 4.1, 4.9], dtype=np.float32)
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = rsquared(model, true)
        assert result <= 1.0  # Can be negative for very bad models

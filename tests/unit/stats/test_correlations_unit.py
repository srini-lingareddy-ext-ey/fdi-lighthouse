"""
Unit tests for statistical correlation functions.

These tests verify the correctness of Pearson, Spearman, and Kendall
correlation calculations.
"""

import numpy as np
import pytest

from lh_v2.shared import ArrayF
from lh_v2.stats.correlations import (
    kendalltau_correlation,
    pearson_correlation,
    spearman_correlation,
)


@pytest.mark.unit
@pytest.mark.stats
class TestPearsonCorrelation:
    """Tests for Pearson correlation calculation."""

    def test_perfect_positive_correlation(self, sample_array: ArrayF):
        """Test that identical arrays have correlation of 1.0."""
        result = pearson_correlation(sample_array, sample_array)
        assert np.isclose(result, 1.0)

    def test_perfect_negative_correlation(self, sample_array: ArrayF):
        """Test that inversely related arrays have correlation of -1.0."""
        inverted = -sample_array
        result = pearson_correlation(sample_array, inverted)
        assert np.isclose(result, -1.0)

    def test_no_correlation(self):
        """Test arrays with no correlation."""
        ts1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ts2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        result = pearson_correlation(ts1, ts2)
        assert np.isnan(result)  # Constant array has undefined correlation

    def test_known_correlation(self):
        """Test with arrays having a known correlation value."""
        ts1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ts2 = np.array([2.0, 4.0, 5.0, 4.0, 5.0], dtype=np.float32)
        result = pearson_correlation(ts1, ts2)
        assert -1.0 <= result <= 1.0

    def test_different_lengths_raises_error(self):
        """Test that arrays of different lengths raise an error."""
        ts1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ts2 = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises((ValueError, IndexError)):
            pearson_correlation(ts1, ts2)


@pytest.mark.unit
@pytest.mark.stats
class TestSpearmanCorrelation:
    """Tests for Spearman correlation calculation."""

    def test_perfect_positive_correlation(self, sample_array: ArrayF):
        """Test that identical arrays have correlation of 1.0."""
        result = spearman_correlation(sample_array, sample_array)
        assert np.isclose(result, 1.0)

    def test_monotonic_relationship(self):
        """Test monotonically increasing relationship."""
        ts1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ts2 = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)  # x^2
        result = spearman_correlation(ts1, ts2)
        assert np.isclose(result, 1.0)  # Perfect rank correlation


@pytest.mark.unit
@pytest.mark.stats
class TestKendallTauCorrelation:
    """Tests for Kendall's tau correlation calculation."""

    def test_perfect_positive_correlation(self, sample_array: ArrayF):
        """Test that identical arrays have correlation of 1.0."""
        result = kendalltau_correlation(sample_array, sample_array)
        assert np.isclose(result, 1.0)

    def test_returns_float(self, sample_array: ArrayF, sample_time_series: ArrayF):
        """Test that the function returns a float."""
        result = kendalltau_correlation(sample_array, sample_array)
        assert isinstance(result, float)

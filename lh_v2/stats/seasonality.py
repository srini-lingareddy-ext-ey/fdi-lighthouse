import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from .linear_regression import std_peacewise_slr_single


def detect_seasonality(ts: np.ndarray, period: int = 12) -> float:
    """
    Return seasonal strength in [0, 1].

    Seasonal strength is defined as ``var(seasonal) / var(ts)``.
    A value close to 1 means the series is dominated by its seasonal
    component; close to 0 means little-to-no seasonality.

    Parameters
    ----------
    ts : np.ndarray
        The time series (1-D) to evaluate.
    period : int
        The seasonal period (12 for monthly data).

    Returns
    -------
    float
        Seasonal strength ∈ [0, 1].
    """
    if len(ts) < 2 * period:
        return 0.0
    result = seasonal_decompose(ts, model='additive', period=period)
    seasonal_var = float(np.var(result.seasonal[~np.isnan(result.seasonal)]))
    total_var = float(np.var(ts))
    if total_var == 0.0:
        return 0.0
    return seasonal_var / total_var


def std_deseasonalized(ts: np.ndarray, n_months_per: int, period: int = 12) -> float:
    """
    Compute piecewise-SLR σ on the deseasonalized (residual + trend)
    component of *ts*.

    If the series is too short to decompose, falls back to the raw σ.
    """
    if len(ts) < 2 * period:
        return std_peacewise_slr_single(ts=ts, n_months_per=n_months_per)
    result = seasonal_decompose(ts, model='additive', period=period)
    deseas = result.trend + result.resid
    # seasonal_decompose pads edges with NaN – strip them
    mask = ~np.isnan(deseas)
    return std_peacewise_slr_single(ts=deseas[mask], n_months_per=n_months_per)

import numpy as np

from lh_v2.datatypes.driver_analysis_types.ranking_types import MAEnum
from lh_v2.shared import ArrayF


def calc_ma(arr_ts: ArrayF, multipliers: ArrayF) -> ArrayF:
    """
    Calculate the moving average of a time series with given multipliers.

    Parameters
    ----------
    arr_ts : np.ndarray
        (shape - (# time series, length of time series))
        A 2D numpy array where each row represents a time series.
    multipliers : np.ndarray
        A 1D numpy array of multipliers to apply to the time series.

    Returns
    -------
    np.ndarray
        A 2D numpy array of the same shape as `ts` containing the moving averages.
    """

    # Initialize output array with same shape as input
    ma_ts = np.zeros(arr_ts.shape)
    # Set first value of moving average to first value of time series
    ma_ts[:, 0] = arr_ts[:, 0]

    # Counter for time steps
    c = 1

    # Handle special cases for early time steps where we don't have enough history
    # to use the full window of multipliers
    while c < multipliers.shape[0]:
        # For each early time step, use as many multipliers as we have data points
        # We reverse the multipliers to apply them chronologically
        ma_ts[:, c] = np.sum(
            arr_ts[:, : c + 1] * multipliers[-c - 1 :], axis=1
        ) / np.sum(multipliers[-c - 1 :])  # Normalize by sum of used multipliers
        c += 1

    # Process the remaining time steps with the full window of multipliers
    for k in range(multipliers.shape[0], arr_ts.shape[1]):
        # Apply the full window of multipliers to the most recent data points
        # For each time step k, we use data points from (k-window+1) to k
        ma_ts[:, k] = np.sum(
            arr_ts[:, k - multipliers.shape[0] + 1 : k + 1] * multipliers, axis=1
        ) / np.sum(multipliers)  # Normalize by sum of all multipliers

    return ma_ts


def ema(arr_ts: ArrayF, a: float) -> ArrayF:
    """
    Compute the Exponential Moving Average (EMA) of a time series.

    Parameters
    ----------
    arr_ts : np.ndarray
        A 2D numpy array where each row represents a different time series.
    a : float
        The smoothing factor for the EMA, where 0 < a <= 1.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing the EMA of the input time series,
        with the same shape as `arr_ts`.

    Notes
    -----
    The EMA is calculated using the formula:
        EMA_t = a * X_t + (1 - a) * EMA_{t-1}
    where X_t is the value at time t, and EMA_t is the EMA at time t.
    """
    # Initialize output array with same shape as input
    arr_out = np.zeros(arr_ts.shape)

    # Set first value of EMA to first value of time series
    # (no previous values to use for smoothing)
    arr_out[:, 0] = arr_ts[:, 0]

    # Calculate EMA for each time step after the first
    for k in range(1, arr_ts.shape[1]):
        # EMA formula: current value * weight + previous EMA * (1-weight)
        arr_out[:, k] = a * arr_ts[:, k] + (1 - a) * arr_out[:, k - 1]

    return arr_out


def moving_average(
    arr_ts: ArrayF,
    ma_type: MAEnum = MAEnum.EXPONENTIAL,
    tail_len: int | None = None,
    a: float = 1 / 2,
):
    """
    Calculate the moving average of a time series using a specified
    type and window length.

    This function computes the moving average of each time series
    in the input array `arr_ts` based on the specified moving average
    type (`ma_type`). The available types are "exponential", "simple",
    and "weighted". The length of the window used for the moving average
    is determined by `tail_len`. For the exponential moving average, a
    decay factor `a` is used to weight the terms.

    Parameters
    ----------
    arr_ts : np.ndarray
        A 2D numpy array where each row represents a time series
        (shape - (# time series, length of time series)).
    ma_type : str, optional
        The type of moving average to calculate.
        Options are "exponential", "simple", and "weighted".
        Default is "exponential".
    tail_len : int, optional
        The length of the window to consider for the moving average.
        Default is -1, which considers the entire length of the time series.
    a : float, optional
        The decay factor for the exponential moving average.
        Default is 1/2.

    Returns
    -------
    np.ndarray
        A 2D numpy array of the same shape as `arr_ts` containing the moving averages.
    """
    if tail_len is None:
        tail_len = int(arr_ts.shape[0])
    multipliers = np.array(list(range(tail_len)))
    if ma_type == MAEnum.EXPONENTIAL:
        return ema(arr_ts, a)
    elif ma_type == MAEnum.SIMPLE:
        multipliers[:] = 1
    elif ma_type == MAEnum.WEIGHTED:
        multipliers += 1
    return calc_ma(arr_ts, multipliers)

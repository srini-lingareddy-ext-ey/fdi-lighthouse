import numpy as np

from lh_v2.shared import ArrayF


def slr(x: ArrayF, y: ArrayF) -> tuple[float, float]:
    """
    Perform simple linear regression.

    Parameters
    ----------
    x : np.ndarray
        The independent variable values (shape - (n,)).
    y : np.ndarray
        The dependent variable values (shape - (n,)).

    Returns
    -------
    tuple
        A tuple (a, b) where 'a' is the intercept and 'b' is the slope of the regression line.
    """
    b = np.sum((x - x.mean()) * (y - y.mean())) / np.sum(np.square(x - x.mean()))
    a = y.mean() - b * x.mean()
    return a, b


def apply_slr(a: float, b: float, x: ArrayF) -> ArrayF:
    """
    Apply simple linear regression (SLR) to the input data.
    Parameters
    ----------
    a : (float)
        The intercept of the linear regression line.
    b : (float)
        The slope of the linear regression line.
    x : (np.ndarray)
        The input data array for which the linear regression is to be applied.

    Returns
    -------
    np.ndarray
        The result of applying the linear regression to the input data.
    """
    return a + b * x


def peacewise_slr(ts: ArrayF, n_months_per: int = 12) -> ArrayF:
    """
    Perform piecewise simple linear regression on a time series.
    This function breaks a time series into segments of a specified length and
    applies simple linear regression to each segment. It then returns the fitted values.

    Parameters
    ----------
    ts : np.ndarray
        The input time series data as a numpy array.
    n_months_per : int, default=12
        The number of time points in each segment for the piecewise regression.

    Returns
    -------
    np.ndarray
        An array of the same shape as the input containing the fitted values
        from the piecewise linear regression.

    Notes
    -----
    The function uses the `slr` and `apply_slr` functions to perform simple
    linear regression on each segment of the time series.
    The last segment may have a different length if the length of the time series
    is not an exact multiple of `n_months_per`.
    """
    # Calculate how many complete segments we can fit into the time series
    n_approx = ts.shape[0] // n_months_per
    # Initialize output array with same shape and data type as input
    arr_approx = np.zeros((ts.shape[0],), ts.dtype)

    # Process each complete segment except the last one
    for k in range(n_approx - 1):
        # Calculate start and end indices for current segment
        start_idx = k * n_months_per
        end_idx = (k + 1) * n_months_per

        # Create x values (time indices) for current segment
        x_values = np.array(list(range(start_idx, end_idx)))
        # Get y values (time series data) for current segment
        y_values = ts[start_idx:end_idx]

        # Calculate slope and intercept using simple linear regression
        slope_intercept = slr(x_values, y_values)

        # Apply the linear regression model to get approximated values
        arr_approx[start_idx:end_idx] = apply_slr(
            *slope_intercept,
            x_values,  # Unpack slope and intercept
        )

    # Handle the remaining data (last segment)
    k = n_approx - 2  # Last segment index
    start_idx = (k + 1) * n_months_per

    # Create x and y values for last segment
    x_values = np.array(list(range(start_idx, ts.shape[0])))
    y_values = ts[start_idx:]

    # Calculate and apply linear regression for last segment
    slope_intercept = slr(x_values, y_values)
    arr_approx[start_idx:] = apply_slr(*slope_intercept, x_values)

    return arr_approx


def std_peacewise_slr_single(ts: ArrayF, n_months_per: int = 20) -> float:
    return float((ts - peacewise_slr(ts=ts, n_months_per=n_months_per)).std())


def std_slr(arr_ts: ArrayF) -> float:
    return float(
        (
            arr_ts
            - apply_slr(
                *slr(np.array(range(len(arr_ts))), arr_ts), np.array(range(len(arr_ts)))
            )
        ).std()
    )


def std_peacewise_slr(arr_ts: ArrayF, n_months_per: int = 20) -> ArrayF:
    return (arr_ts - np.apply_along_axis(peacewise_slr, 1, arr_ts, n_months_per)).std(
        axis=1
    )

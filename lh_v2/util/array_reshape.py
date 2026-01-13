import numpy as np

from lh_v2.shared import ArrayF


def reshape_data_n_months(
    arr_x: ArrayF, arr_y: ArrayF | None = None, n_months_used: int = 3
) -> tuple[ArrayF, ArrayF | None]:
    """
    Formats input data for use with XGBoost models.

    This function transforms the input driver data (`arr_x`)
    and optionally the target data (`arr_y`) into a format
    suitable for training or prediction with XGBoost.
    It uses a sliding window approachto include data from
    the specified number of past months (`n_months_used`).

    Parameters
    ----------
    arr_x : np.ndarray
        A 2D numpy array where each row represents a driver and
        each column represents a time period.
    arr_y : np.ndarray or None, optional
        A 1D numpy array containing the target values (e.g., account data)
        corresponding to `arr_x`.
        If None, only the transformed `arr_x` is returned. Default is None.
    n_months_used : int, optional
        The number of past months to include in the transformation. Default is 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray or None]
        A tuple containing:
        - arr_x_out : np.ndarray
            A 2D numpy array where each row represents the transformed
            driver data for a specific time period.
        - arr_y_out : np.ndarray or None
            A 1D numpy array containing the target values corresponding
            to the transformed driver data.
            If `arr_y` is None, this will also be None.

    Notes
    -----
    - The number of rows in `arr_x_out` will be reduced by `n_months_used - 1`
        compared to the original `arr_x`.
    - The transformation flattens the data for the specified number of past months
        into a single row for each time period.
    """
    if n_months_used == 1:
        return arr_x.T, arr_y
    arr_x_out = np.zeros(
        (arr_x.shape[1] - (n_months_used - 1), arr_x.shape[0] * n_months_used),
        arr_x.dtype,
    )
    if arr_y is not None:
        arr_y_out = arr_y[n_months_used - 1 :]
    else:
        arr_y_out = None
    for k in np.arange(arr_x_out.shape[0]):
        arr_x_out[k] = arr_x[:, k : k + n_months_used].flatten()
    return arr_x_out, arr_y_out

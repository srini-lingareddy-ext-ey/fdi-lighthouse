import numpy as np

from lh_v2.shared import ArrayF


def normalize(time_series: ArrayF) -> ArrayF:
    """
    Function that returnes a normalized (mean 0, sd 1) version of the given time series

    Parameters
    ----------
    time_series : (np.ndarray)
        (shape - (length of time series,)) the vector to be stats normalized.

    Returns
    -------
    np.ndarray
        (shape - (length of time series,)) the stats normalized vector
    """
    return (time_series - time_series.mean()) / time_series.std()


def normalize_arr(arr: ArrayF) -> ArrayF:
    """
    Function that returnes a normalized (mean 0, sd 1) version of the given array
    The assumed shape of the array is (# time series, len time series)

    Parameters
    ----------
    arr : (np.ndarray)
        (shape - (# of time series, length of time series))
        the time series that need to be normalized

    Returns
    -------
    np.ndarray
        (shape - (# of time series, length of time series))
        the stats normalized time series
    """
    return ((arr.T - arr.mean(axis=1)) / arr.std(axis=1)).T


def len_norm(vector: ArrayF) -> ArrayF:
    """
    Function to normalize the length of a vector

    Parameters
    ----------
    vector : (np.ndarray)
        (shape - (length of vector,))
        the vector to be length normalized.

    Returns
    -------
    np.ndarray
        (shape - (length of vector,))
        the length normed vector
    """
    return vector / (np.sqrt(np.sum(np.square(vector))))


def len_norm_arr(arr: ArrayF) -> ArrayF:
    """
    Function to normalize the length of the columns of arr.
    The assumed shape of the array is (# vectors, len vectors)

    Parameters
    ----------
    arr : (np.ndarray)
        (shape - (# vectors, length of vectors))
        the vectors to be length normalized.

    Returns
    -------
    np.ndarray
        (shape - (# vectors, length of vector))
        the length normed vectors
    """
    return (arr.T / np.sqrt(np.sum(np.square(arr), axis=1))).T


def std_1d(arr_ts: ArrayF) -> float:
    return float(arr_ts.std())


def std_basic(arr_ts: ArrayF) -> ArrayF:
    return arr_ts.std(axis=1)


def std_difference(arr_ts: ArrayF) -> ArrayF:
    diff = arr_ts[:, 1:] - arr_ts[:, :-1]
    return diff.std(axis=1)

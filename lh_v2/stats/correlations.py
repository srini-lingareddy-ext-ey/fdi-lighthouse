import numpy as np
import scipy.stats

from lh_v2.shared import ArrayF


def pearson_correlation(ts1: ArrayF, ts2: ArrayF) -> float:
    """
    Calculate the Pearson correlation coefficient between two time series.

    Parameters
    ----------
    ts1 : np.ndarray
        The first time series (shape - (n,)).
    ts2 : np.ndarray
        The second time series (shape - (n,)).

    Returns
    -------
    float
        The Pearson correlation coefficient between the two time series.
    """
    return ((ts1 - ts1.mean()) / ts1.std()).dot(
        (ts2 - ts2.mean()) / ts2.std()
    ) / ts1.shape[0]


def spearman_correlation(ts1: ArrayF, ts2: ArrayF) -> float:
    """
    Calculate the Spearman rank correlation coefficient between two time series.

    Parameters
    ----------
    ts1 : np.ndarray
        The first time series (shape - (n,)).
    ts2 : np.ndarray
        The second time series (shape - (n,)).

    Returns
    -------
    float
        The Spearman rank correlation coefficient between the two time series.
    """
    ts1_ranking = np.argsort(ts1)
    ts2_ranking = np.argsort(ts2)
    return 1 - (
        6
        * np.sum(np.square(ts1_ranking - ts2_ranking))
        / (ts1.shape[0] * (ts1.shape[0] ** 2 - 1))
    )


def kendalltau_correlation(ts1: ArrayF, ts2: ArrayF) -> float:
    """
    Calculate the Kendall Tau correlation coefficient between two time series.

    Parameters
    ----------
    ts1 : np.ndarray
        The first time series (shape - (n,)).
    ts2 : np.ndarray
        The second time series (shape - (n,)).

    Returns
    -------
    float
        The Kendall Tau correlation coefficient between the two time series.
    """
    return float(scipy.stats.kendalltau(ts1, ts2).statistic)  # type: ignore

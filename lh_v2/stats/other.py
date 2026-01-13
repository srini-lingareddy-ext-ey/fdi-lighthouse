import numpy as np

from lh_v2.shared import ArrayF


def kl_divergence(p: ArrayF, q: ArrayF) -> float:
    """
    Calculate the Kullback-Leibler (KL) divergence
    between two probability distributions.
    The KL divergence is a measure of how one probability distribution
    diverges from a second, expected probability distribution.

    Parameters
    ----------
    p : np.ndarray
        The first probability distribution.
    q : np.ndarray
        The second probability distribution.

    Returns
    -------
    float
        The KL divergence between the two distributions.
    """
    return float(np.sum(p * np.log(p / q)))


def gamma(best_lag: int, n_months_predicted: int) -> float:
    if best_lag == n_months_predicted // 2:
        return 1
    elif best_lag < n_months_predicted // 2:
        return 0.5 / (
            n_months_predicted // 2 - best_lag + best_lag / n_months_predicted
        )
    else:
        return best_lag - n_months_predicted // 2 + best_lag / n_months_predicted


def power_transform(
    value_low: float, value_high: float, best_lag: int, n_months_predicted: int
) -> ArrayF:
    arr = np.linspace(0, 1, n_months_predicted + 1)[1:] ** gamma(
        best_lag, n_months_predicted
    )
    return arr * (value_high - value_low) + value_low

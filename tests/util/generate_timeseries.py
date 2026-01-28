import numpy as np

from lh_v2.shared import BASE_NP_DTYPE, ArrayF, ArrayI


def breakup_lengths(
    total_length: int,
    n_segments: int,
    max_variance: int = 3,
) -> ArrayI:
    """Break up total_length into n_segments with some variance."""
    if n_segments < 1:
        n_segments = total_length // 20
    elif n_segments > total_length // 8:
        n_segments = total_length // 10

    segment_variance = np.random.randint(
        -max_variance, max_variance + 1, size=n_segments
    )
    segment_lengths = np.zeros(n_segments, dtype=int)
    segment_lengths[0] = total_length // n_segments + segment_variance[0]
    for i in range(1, n_segments):
        segment_lengths[i] = (
            total_length // n_segments + segment_variance[i] - segment_variance[i - 1]
        )
    segment_lengths[-1] = total_length - np.sum(segment_lengths[:-1])

    return segment_lengths


def generate_slopes(
    n_segments: int,
    slope_mean: float,
    slope_std: float,
    np_dtype: type = BASE_NP_DTYPE,
) -> ArrayF:
    """Generate random slopes for each segment."""
    slopes = np.random.normal(loc=slope_mean, scale=slope_std, size=n_segments)
    return slopes.astype(np_dtype)


# Fun idea, but would require too much time to implement properly
# This is a hard problem
def generate_sample_ts(
    len_ts: int,
    n_breaks: int,
    slope_mean: float = 0,
    slope_std: float = 0.01,
    noise_std: float = 0.1,
    b_noise: bool = True,
    b_center: bool = True,
    np_dtype: type = BASE_NP_DTYPE,
) -> ArrayF:
    """Generate a sample time series array for testing."""
    if n_breaks < 1:
        n_breaks = 1
    elif n_breaks > len_ts // 8:
        n_breaks = len_ts // 10

    segment_lengths = breakup_lengths(total_length=len_ts, n_segments=n_breaks + 1)
    n_breaks = segment_lengths.shape[0] - 1
    slopes = generate_slopes(
        n_segments=n_breaks + 1,
        slope_mean=slope_mean,
        slope_std=slope_std,
        np_dtype=np_dtype,
    )
    ts = np.zeros(len_ts, dtype=np_dtype)
    start_val = 10.0
    start_idx = 0
    for idx in range(n_breaks + 1):
        ts[start_idx : start_idx + segment_lengths[idx]] = (
            start_val + slopes[idx] * np.arange(1, segment_lengths[idx] + 1)
        ).astype(np_dtype)
        start_val = ts[start_idx + segment_lengths[idx] - 1]
        start_idx += segment_lengths[idx]

    if b_noise:
        ts += np.random.normal(loc=0.0, scale=noise_std, size=len_ts).astype(np_dtype)

    if b_center:
        ts -= ts.mean()  # Center around 0
    return ts

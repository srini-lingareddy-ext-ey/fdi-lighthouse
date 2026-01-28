import matplotlib.pyplot as plt
import numpy as np

from lh_v2.shared import BASE_NP_DTYPE, ArrayF
from tests.util.generate_timeseries import generate_sample_ts


def plot_generated_timeseries(
    time_series: ArrayF,
):
    """Plot the generated time series for visual inspection."""
    plt.figure(figsize=(15, 9))
    plt.plot(time_series.T)
    plt.title('Generated Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    return


if __name__ == '__main__':
    n_ts = 100
    len_ts = 80
    n_breaks = 5
    ts = np.zeros((n_ts, len_ts), dtype=BASE_NP_DTYPE)
    for i in range(ts.shape[0]):
        ts[i] = generate_sample_ts(
            len_ts=ts.shape[1],
            n_breaks=n_breaks,
            slope_mean=0.5,
            slope_std=0.5,
            b_noise=False,
            b_center=False,
        )

    plot_generated_timeseries(ts)

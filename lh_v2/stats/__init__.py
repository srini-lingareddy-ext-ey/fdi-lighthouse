from .correlations import (
    kendalltau_correlation,
    pearson_correlation,
    spearman_correlation,
)
from .lin_alg import least_squares, pca, subspace_proj
from .linear_regression import (
    apply_slr,
    peacewise_slr,
    slr,
    std_peacewise_slr,
    std_peacewise_slr_single,
    std_slr,
)
from .moving_averages import calc_ma, ema, moving_average
from .norm_error import (
    l1_norm,
    l2_norm,
    mape,
    mse,
    mse_percentage,
    rmse,
    rmse_percentage,
    rsquared,
)
from .normalize_arrs import (
    len_norm,
    len_norm_arr,
    normalize,
    normalize_arr,
    std_1d,
    std_basic,
    std_difference,
)
from .other import gamma, kl_divergence, power_transform
from .seasonality import detect_seasonality, std_deseasonalized

__all__ = [
    'kendalltau_correlation',
    'pearson_correlation',
    'spearman_correlation',
    'least_squares',
    'pca',
    'subspace_proj',
    'apply_slr',
    'peacewise_slr',
    'slr',
    'std_slr',
    'std_peacewise_slr_single',
    'std_peacewise_slr',
    'calc_ma',
    'ema',
    'moving_average',
    'l1_norm',
    'l2_norm',
    'mape',
    'mse',
    'mse_percentage',
    'rmse',
    'rmse_percentage',
    'rsquared',
    'len_norm',
    'len_norm_arr',
    'normalize',
    'normalize_arr',
    'std_1d',
    'std_basic',
    'std_difference',
    'detect_seasonality',
    'std_deseasonalized',
    'gamma',
    'kl_divergence',
    'power_transform',
]

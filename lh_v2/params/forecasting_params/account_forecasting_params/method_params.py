from typing import Any, Literal, Sequence

from lh_v2.util import BaseParamsModel


class BaseAccountForecastParams(BaseParamsModel):
    def make_all_opt_params(
        self,
        dict_fields_idx: dict[str, int],
        params_lst: Sequence[Sequence[int | float]],
    ) -> list[BaseAccountForecastParams]:
        fields = {f for f in type(self).model_fields.keys()}
        new_fields: list[dict[str, Any]] = [{} for _ in range(len(params_lst))]
        for field in fields:
            if field in dict_fields_idx.keys():
                for i in range(len(params_lst)):
                    new_fields[i][field] = params_lst[i][dict_fields_idx[field]]
            else:
                for i in range(len(params_lst)):
                    new_fields[i][field] = getattr(self, field)

        return [type(self)(**fields) for fields in new_fields]


class BaseAccountForecastParamsRange(BaseParamsModel):
    pass


class LinearRegressionAccountForecastParams(BaseAccountForecastParams):
    pass


class LinearRegressionAccountForecastParamsRange(BaseAccountForecastParamsRange):
    pass


class MovingAverageAccountForecastParams(BaseAccountForecastParams):
    window_size: int = 3  # Number of periods to average


class MovingAverageAccountForecastParamsRange(BaseAccountForecastParamsRange):
    # 3-12 months is reasonable for monthly data
    window_size_range: tuple[int, int] = (3, 12)


class ExponentialSmoothingAccountForecastParams(BaseAccountForecastParams):
    trend: str | None = None  # Trend type: 'add', 'mul', or None (auto-detect)
    seasonal: str | None = None  # Seasonal type: 'add', 'mul', or None (auto-detect)
    seasonal_periods: int = 12  # 12 for monthly data
    auto_detect: bool = True  # Auto-detect trend/seasonality
    seasonality_threshold: float = 0.1  # Threshold for seasonal strength (10%)


class ExponentialSmoothingAccountForecastParamsRange(BaseAccountForecastParamsRange):
    # Categorical parameters (trend, seasonal) cannot be tuned
    seasonality_threshold_range: tuple[float, float] = (0.05, 0.2)


class ProphetAccountForecastParams(BaseAccountForecastParams):
    seasonality_mode: str = 'additive'  # 'additive' or 'multiplicative'
    yearly_seasonality: bool | int | str = (
        'auto'  # Yearly seasonality (auto/True/False/int)
    )
    weekly_seasonality: bool | str = (
        False  # Weekly seasonality (usually False for monthly)
    )
    daily_seasonality: bool | str = (
        False  # Daily seasonality (usually False for monthly)
    )
    changepoint_prior_scale: float = 0.05  # Trend flexibility (higher = more flexible)
    seasonality_prior_scale: float = (
        10.0  # Seasonality flexibility (higher = more flexible)
    )


class ProphetAccountForecastParamsRange(BaseAccountForecastParamsRange):
    # Note: Categorical parameter (seasonality_mode) cannot be tuned
    changepoint_prior_scale_range: tuple[float, float] = (0.01, 0.5)
    seasonality_prior_scale_range: tuple[float, float] = (1.0, 20.0)


class RandomForestAccountForecastParams(BaseAccountForecastParams):
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42  # Fixed for reproducibility - never tune
    n_jobs: int = 1  # Number of parallel threads (-1 = use all cores)


class RandomForestAccountForecastParamsRange(BaseAccountForecastParamsRange):
    n_estimators_range: tuple[int, int] = (50, 300)  # Number of trees in the forest
    max_depth_range: tuple[int, int] = (3, 20)  # Maximum depth of the tree
    min_samples_split_range: tuple[int, int] = (
        2,
        10,
    )  # Min samples to split: higher = more conservative, less overfitting
    min_samples_leaf_range: tuple[int, int] = (
        1,
        5,
    )  # Min samples per leaf: higher = smoother predictions, less overfitting


class XGBoostAccountForecastParams(BaseAccountForecastParams):
    n_estimators: int = 100  # Number of boosting rounds
    max_depth: int = 3  # Maximum tree depth for base learners
    learning_rate: float = 0.1  # Step size shrinkage
    subsample: float = 0.8  # Fraction of training data for each tree
    colsample_bytree: float = 0.8  # Fraction of features for each tree
    reg_alpha: float = 0.0  # L1 regularization term on weights
    reg_lambda: float = 1.0  # L2 regularization term on weights
    random_state: int = 42  # Fixed for reproducibility - never tune
    n_jobs: int = 1  # Number of parallel threads (-1 = use all cores)


class XGBoostAccountForecastParamsRange(BaseAccountForecastParamsRange):
    # Wider ranges for better hyperparameter optimization
    n_estimators_range: tuple[int, int] = (10, 600)  # Number of boosting rounds
    max_depth_range: tuple[int, int] = (1, 6)  # Maximum tree depth for base learners
    learning_rate_range: tuple[float, float] = (0.005, 0.5)  # Step size shrinkage
    reg_lambda_range: tuple[float, float] = (0.0, 15.0)  # L2 regularization
    subsample_range: tuple[float, float] = (
        0.05,
        1.0,
    )  # Fraction of training data for each tree
    colsample_bytree_range: tuple[float, float] = (
        0.05,
        1.0,
    )  # Fraction of features for each tree
    reg_alpha_range: tuple[float, float] = (0.0, 15.0)  # L1 regularization

    # Previous narrower production ranges (smaller but faster):
    # n_estimators_range: tuple[int, int] = (10, 500)
    # max_depth_range: tuple[int, int] = (1, 4)
    # learning_rate_range: tuple[float, float] = (0.01, 0.5)
    # reg_lambda_range: tuple[float, float] = (0.0, 10.0)
    # subsample_range: tuple[float, float] = (0.25, 0.95)
    # colsample_bytree_range: tuple[float, float] = (0.6, 1.0)
    # reg_alpha_range: tuple[float, float] = (0.0, 10.0)


class RidgeAccountForecastParams(BaseAccountForecastParams):
    alpha: float = 1.0  # Regularization strength
    fit_intercept: bool = True
    solver: Literal['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] = (
        'auto'  # 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
    )
    random_state: int = 42  # Fixed for reproducibility - never tune


class RidgeAccountForecastParamsRange(BaseAccountForecastParamsRange):
    alpha_range: tuple[float, float] = (0.1, 10.0)  # Regularization strength


class LassoAccountForecastParams(BaseAccountForecastParams):
    alpha: float = 1.0  # Regularization strength (L1 penalty)
    fit_intercept: bool = True
    max_iter: int = 1000  # High enough to ensure convergence on most datasets
    tol: float = 1e-4  # Tolerance for optimization
    selection: Literal['cyclic', 'random'] = (
        'cyclic'  # 'cyclic' or 'random' for feature selection order
    )
    random_state: int = 42  # Fixed for reproducibility - never tune


class LassoAccountForecastParamsRange(BaseAccountForecastParamsRange):
    alpha_range: tuple[float, float] = (
        0.1,
        10.0,
    )  # Lower = less regularization (more features), higher = more regularization (fewer features)


class HyperLassoAccountForecastParams(BaseAccountForecastParams):
    alphas: list[float] | int = 100  # Alpha values to try (None = auto-generate)
    cv: int = 5  # Number of cross-validation folds
    fit_intercept: bool = True
    max_iter: int = 1000  # High enough to ensure convergence on most datasets
    tol: float = 1e-4  # Tolerance for optimization
    n_jobs: int = 1  # Number of parallel jobs (-1 = use all cores)
    random_state: int | None = 42


class HyperLassoAccountForecastParamsRange(BaseAccountForecastParamsRange):
    # auto-tunes alpha internally via cross-validation
    # Could tune cv (number of folds) but 5 is a strong standard
    pass


class SARIMAXAccountForecastParams(BaseAccountForecastParams):
    seasonal: bool = True  # Whether to include seasonal component
    auto_params: bool = True  # Auto-search for optimal parameters (slow)
    default_params: list[int] = [1, 1, 1, 0, 0, 0, 12]  # (p,d,q,P,D,Q,s)
    max_params: list[int] = [2, 1, 2, 1, 1, 1, 12]  # Max search ranges for auto


class SARIMAXAccountForecastParamsRange(BaseAccountForecastParamsRange):
    pass

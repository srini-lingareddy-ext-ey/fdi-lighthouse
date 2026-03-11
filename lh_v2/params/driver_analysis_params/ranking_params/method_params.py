from lh_v2.datatypes.driver_analysis_types.ranking_types import (
    MAEnum,
    PCADimMethods,
)
from lh_v2.util import BaseParamsModel


class BaseRankingParams(BaseParamsModel):
    pass


class PearsonRankingParams(BaseRankingParams):
    pass


class SpearmanRankingParams(BaseRankingParams):
    pass


class SLRRankingParams(BaseRankingParams):
    pass


class MARankingParams(BaseRankingParams):
    """
    Parameters for Moving Average (MA) ranking.

    Attributes
    ----------
    tail_len : int
        The length of the tail to consider for the moving average calculation.
    ma_type : str
        The type of moving average to use. Options include "exponential".
    a : float
        The smoothing factor for the moving average, typically between 0 and 1.
    """

    tail_len: int = 6
    """ The length of the tail to consider for the moving average calculation."""
    ma_type: MAEnum = MAEnum.EXPONENTIAL
    """The type of moving average to use. Options include "exponential"."""
    a: float = 0.5
    """The smoothing factor for the moving average, typically between 0 and 1."""


class PCARankingParams(BaseRankingParams):
    """
    Parameters for PCA-based ranking.

    Attributes
    ----------
    dim_method : PCADimMethods
        The dimensionality reduction method to be used. Default is `PCADimMethods.ACCOUNT`.
    """

    dim_method: PCADimMethods = PCADimMethods.ACCOUNT


class RFERankingParams(BaseRankingParams):
    """
    Parameters for Recursive Feature Elimination (RFE) ranking.

    Attributes
    ----------
    n_iter : int
        The number of iterations to perform during the RFE process.
    n_estimators : int
        The number of estimators to use in the underlying model for feature ranking.
    """

    n_iter: int = 50
    n_estimators: int = 100


class RFELinearRegressionRankingParams(BaseRankingParams):
    """
    Parameters for Recursive Feature Elimination with Linear Regression ranking.

    Attributes
    ----------
    n_iter : int
        The number of iterations to perform during the RFE process.
    """

    n_iter: int = 50


class LassoRankingParams(BaseRankingParams):
    """
    Parameters for configuring the Lasso ranking model.

    Attributes
    ----------
    n_iter : int
        The number of iterations to perform during the ranking process.
        Default is 100.
    max_iter : int
        The maximum number of iterations for the Lasso solver.
        Default is 100.
    alpha : float
        Regularization strength for Lasso regression. Higher values specify
        stronger regularization. Default is 0.1.
    """

    n_iter: int = 100
    max_iter: int = 50
    alpha: float = 0.5


class RidgeRankingParams(BaseRankingParams):
    """
    Parameters for configuring the Ridge regression ranking model.

    Attributes
    ----------
    n_iter : int
        The number of iterations to perform during the ranking process.
        Default is 80.
    alpha : float
        Regularization strength for Ridge regression. Higher values specify
        stronger regularization. Default is 1.0.
    """

    n_iter: int = 80
    alpha: float = 1.0


class XGBoostRankingParams(BaseRankingParams):
    """
    Parameters for configuring the XGBoost ranking model.

    Attributes
    ----------
    n_iter : int
        The number of iterations to perform during the ranking process.
        Default is 80.
    n_estimators : int
        The number of boosting rounds. Default is 100.
    max_depth : int
        Maximum tree depth for base learners. Default is 3.
    learning_rate : float
        Boosting learning rate (step size shrinkage). Default is 0.1.
    subsample : float
        Subsample ratio of the training instances. Default is 0.8.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree. Default is 0.8.
    """

    n_iter: int = 10
    n_estimators: int = 60
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class BorutaRankingParams(BaseRankingParams):
    """
    Parameters for configuring the Boruta feature selection ranking model.

    Attributes
    ----------
    max_iter : int
        Maximum number of Boruta iterations. Default is 100.
    n_estimators : int
        Number of trees in Random Forest for importance calculation. Default is 100.
    perc : int
        Percentile for determining shadow feature importance threshold (0-100).
        Default is 100 (uses maximum shadow importance).
    alpha : float
        Significance level for statistical test. Default is 0.05.
    max_depth : int
        Maximum tree depth for Random Forest. Default is 5.
    """

    max_iter: int = 100
    n_estimators: int = 100
    perc: int = 100
    alpha: float = 0.05
    max_depth: int = 5


class RandomForestLightGBMRankingParams(BaseRankingParams):
    """
    Parameters for configuring the LightGBM Random Forest ranking model.

    Attributes
    ----------
    n_estimators : int
        Number of trees in the random forest. Default is 100.
    min_child_samples : int
        Minimum number of samples required in a leaf node. Default is 2.
    subsample : float
        Subsample ratio of training instances for bagging (0.0 to 1.0). Default is 0.6.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree (0.0 to 1.0). Default is 0.6.
    random_state : int
        Random seed for reproducibility. Default is 42.
    """

    n_estimators: int = 100
    min_child_samples: int = 2
    subsample: float = 0.6
    colsample_bytree: float = 0.6
    random_state: int = 42


class MRMRRankingParams(BaseRankingParams):
    """
    Parameters for configuring the mRMR (minimum Redundancy Maximum Relevance) ranking model.

    Attributes
    ----------
    n_features : int
        Number of top features to select. Default is 10.
    n_bins : int
        Number of bins for discretizing continuous variables when computing
        mutual information. Default is 10.
    """

    n_features: int = 10
    n_bins: int = 10

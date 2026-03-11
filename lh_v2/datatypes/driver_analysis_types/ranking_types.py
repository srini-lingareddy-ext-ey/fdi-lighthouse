from enum import Enum


class PCADimMethods(Enum):
    """
    Enumeration for PCA Dimension Methods.

    This enumeration defines the methods for determining the dimensions
    used in Principal Component Analysis (PCA) for driver ranking.

    Attributes
    ----------
    ACCOUNT : str
        Use the account data to determine the dimensions for PCA.
    DRIVERS : str
        Use the driver data to determine the dimensions for PCA.
    """

    ACCOUNT = 'account'
    DRIVERS = 'drivers'


class MAEnum(Enum):
    """Enumeration for Moving Average types.

    This enum defines the types of moving averages that can be used in calculations.

    Attributes
    ----------
    EXPONENTIAL : str
        Exponential moving average, which gives more weight to recent data points.
    SIMPLE : str
        Simple moving average, which gives equal weight to all data points in the window.
    WEIGHTED : str
        Weighted moving average, which assigns different weights to data points
        based on their position in the window.
    """

    EXPONENTIAL = 'exponential'
    SIMPLE = 'simple'
    WEIGHTED = 'weighted'


class DriverRankingEnum(Enum):
    """
    DriverRankingEnum is an enumeration that defines various methods for driver ranking.

    Attributes
    ----------
    PEARSON_CORRELATION : str
        Represents the Pearson correlation method for ranking drivers.
    SPEARMAN_CORRELATION : str
        Represents the Spearman correlation method for ranking drivers.
    SLR : str
        Represents the Simple Linear Regression (SLR) method for ranking drivers.
    MOVING_AVERAGE : str
        Represents the Moving Average method for ranking drivers.
    PCA : str
        Represents the Principal Component Analysis (PCA) method for ranking drivers.
    RFE : str
        Represents the Recursive Feature Elimination (RFE) method for ranking drivers (Random Forest).
    RFE_LINEAR_REGRESSION : str
        Represents the Recursive Feature Elimination with Linear Regression base estimator for ranking drivers.
    LASSO : str
        Represents the Least Absolute Shrinkage and Selection Operator (LASSO) method for ranking drivers.
    RIDGE : str
        Represents the Ridge regression method for ranking drivers.
    XGBOOST : str
        Represents the XGBoost (Extreme Gradient Boosting) method for ranking drivers.
    BORUTA : str
        Represents the Boruta feature selection method for ranking drivers.
    RANDOM_FOREST_LIGHTGBM : str
        Represents the Random Forest method using LightGBM implementation for ranking drivers.
    MRMR : str
        Represents the mRMR (minimum Redundancy Maximum Relevance) method for ranking drivers.
    """

    PEARSON_CORRELATION = 'pearson_correlation'
    SPEARMAN_CORRELATION = 'spearman_correlation'
    SLR = 'slr'
    MOVING_AVERAGE = 'moving_average'
    PCA = 'pca'
    RFE = 'rfe'
    RFE_LINEAR_REGRESSION = 'rfe_linear_regression'
    LASSO = 'lasso'
    RIDGE = 'ridge'
    XGBOOST = 'xgboost'
    BORUTA = 'boruta'
    RANDOM_FOREST_LIGHTGBM = 'random_forest_lightgbm'
    MRMR = 'mrmr'


class DriverRankingMetric(Enum):
    """
    Enumeration for LLM-based Driver Ranking Metrics.

    This enumeration defines the metrics used to evaluate and rank drivers
    in the context of LLM-based driver analysis.

    Attributes
    ----------
    FINAL_RANK : str
        Represents the final ranking metric for ranking drivers.
    AVG_RANK : str
        Represents the average ranking metric for ranking drivers.
    PEARSON_CORRELATION : str
        Represents the Pearson correlation metric for ranking drivers.
    """

    FINAL_RANK = 'final_rank'
    AVG_RANK = 'avg_rank'
    PEARSON_CORRELATION = DriverRankingEnum.PEARSON_CORRELATION.value
    SPEARMAN_CORRELATION = DriverRankingEnum.SPEARMAN_CORRELATION.value
    SLR = DriverRankingEnum.SLR.value
    MOVING_AVERAGE = DriverRankingEnum.MOVING_AVERAGE.value
    PCA = DriverRankingEnum.PCA.value
    RFE = DriverRankingEnum.RFE.value
    RFE_LINEAR_REGRESSION = DriverRankingEnum.RFE_LINEAR_REGRESSION.value
    LASSO = DriverRankingEnum.LASSO.value
    RIDGE = DriverRankingEnum.RIDGE.value
    XGBOOST = DriverRankingEnum.XGBOOST.value
    BORUTA = DriverRankingEnum.BORUTA.value
    RANDOM_FOREST_LIGHTGBM = DriverRankingEnum.RANDOM_FOREST_LIGHTGBM.value
    MRMR = DriverRankingEnum.MRMR.value

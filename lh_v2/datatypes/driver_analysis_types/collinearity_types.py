from enum import Enum


class CollinearityMethodEnum(Enum):
    """
    CollinearityMethodEnum is an enumeration that defines methods for assessing collinearity.

    Attributes
    ----------
    CORRELATION : str
        Represents the correlation-based method for evaluating collinearity.
    VIF : str
        Represents the Variance Inflation Factor (VIF) method for assessing collinearity.
    KMEANS : str
        Represents the k-means clustering method for analyzing collinearity.
    MUTUAL_INFORMATION : str
        Represents the mutual information method for assessing collinearity.
    LASSO : str
        Represents the LASSO regression coefficient method for detecting collinearity.
    HIERARCHICAL_CLUSTERING : str
        Represents the hierarchical clustering method for detecting collinearity.
    """

    CORRELATION = 'correlation'
    VIF = 'vif'
    KMEANS = 'kmeans'
    MUTUAL_INFORMATION = 'mutual_information'
    LASSO = 'lasso'
    HIERARCHICAL_CLUSTERING = 'hierarchical_clustering'

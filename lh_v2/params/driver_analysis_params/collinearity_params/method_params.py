from typing import Literal

from lh_v2.util import BaseParamsModel


class BaseCollinearityParams(BaseParamsModel):
    pass


class CorrelationCollinearityParams(BaseCollinearityParams):
    """
    Parameters for managing collinearity using correlation metrics.

    This class defines weights for three different correlation metrics:
    Pearson, Spearman, and Kendall. These weights are used to balance
    the contribution of each metric when assessing collinearity.

    Attributes
    ----------
    pearson_weight : float
        The weight assigned to the Pearson correlation coefficient. Default is 1/3.
    spearman_weight : float
        The weight assigned to the Spearman rank correlation coefficient. Default is 1/3.
    kendall_weight : float
        The weight assigned to the Kendall rank correlation coefficient. Default is 1/3.
    """

    pearson_weight: float = 1 / 3
    spearman_weight: float = 1 / 3
    kendall_weight: float = 1 / 3


class VIFCollinearityParams(BaseCollinearityParams):
    pass


class KMeansCollinearityParams(BaseCollinearityParams):
    """
    Parameters for configuring KMeans-based collinearity analysis.

    Attributes
    ----------
    n_seeds : int
        The number of random seeds to use for initializing the KMeans algorithm.
        This determines the number of different initializations to evaluate.
    """

    n_seeds: int = 40


class MICollinearityParams(BaseCollinearityParams):
    """
    Parameters for configuring Mutual Information-based collinearity analysis.

    Attributes
    ----------
    n_bins : int
        The number of bins to use for histogram discretization when computing
        mutual information between driver pairs. Default is 10.
    """

    n_bins: int = 10


class LassoCollinearityParams(BaseCollinearityParams):
    """
    Parameters for configuring LASSO-based collinearity analysis.

    Attributes
    ----------
    alpha : float
        The regularization parameter for LASSO regression. Higher values
        lead to more sparsity (fewer non-zero coefficients). Default is 0.01.
    """

    alpha: float = 0.01


class HierarchicalClusteringCollinearityParams(BaseCollinearityParams):
    """
    Parameters for configuring hierarchical clustering-based collinearity analysis.

    Attributes
    ----------
    linkage_method : str
        The linkage method for hierarchical clustering. Options are 'ward',
        'complete', 'average', or 'single'. Default is 'ward'.
    distance_metric : str
        The distance metric to use. Options are 'correlation' or 'euclidean'.
        Default is 'correlation'.
    """

    linkage_method: Literal['ward', 'complete', 'average', 'single'] = 'ward'
    distance_metric: Literal['correlation', 'euclidean'] = 'correlation'

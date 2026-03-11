from typing import Sequence

from pydantic import Field

from lh_v2.datatypes.driver_analysis_types.collinearity_types import (
    CollinearityMethodEnum,
)
from lh_v2.util import BaseParamsModel

from .method_params import (
    BaseCollinearityParams,
    CorrelationCollinearityParams,
    HierarchicalClusteringCollinearityParams,
    KMeansCollinearityParams,
    LassoCollinearityParams,
    MICollinearityParams,
    VIFCollinearityParams,
)


class CollinearityParams(BaseParamsModel):
    """
    Parameters for handling collinearity in data.

    This class encapsulates the configuration for collinearity removal and selection
    methods, as well as their associated parameters.

    Attributes
    ----------
    b_remove : bool
        Indicates whether collinearity removal is enabled.
    methods_removed : Sequence[CollinearityMethodEnum]
        A sequence of collinearity removal methods to be applied.
    b_selected : bool
        Indicates whether collinearity selection is enabled.
    methods_selected : Sequence[CollinearityMethodEnum]
        A sequence of collinearity selection methods to be applied.
    correlation_collinearity_params : CorrelationCollinearityParams
        Parameters specific to the correlation-based collinearity method.
    vif_collinearity_params : VIFCollinearityParams
        Parameters specific to the Variance Inflation Factor (VIF) collinearity method.
    kmeans_collinearity_params : KMeansCollinearityParams
        Parameters specific to the K-Means collinearity method.
    mi_collinearity_params : MICollinearityParams
        Parameters specific to the Mutual Information collinearity method.
    lasso_collinearity_params : LassoCollinearityParams
        Parameters specific to the LASSO collinearity method.
    hierarchical_clustering_collinearity_params : HierarchicalClusteringCollinearityParams
        Parameters specific to the Hierarchical Clustering collinearity method.

    Methods
    -------
    __getitem__(key: CollinearityMethodEnum)
        Retrieves the parameters associated with the specified collinearity method.

    Raises
    ------
    ValueError
        If the provided key is not a valid collinearity method of type CollinearityMethodEnum.
    """

    b_remove: bool = False
    methods_removed: Sequence[CollinearityMethodEnum] = ()
    b_selected: bool = False
    methods_selected: Sequence[CollinearityMethodEnum] = ()
    threashold_base_removal: float = 0.7
    threashold_base_pruning: float = 0.85
    correlation_collinearity_params: CorrelationCollinearityParams = Field(
        default_factory=CorrelationCollinearityParams
    )
    vif_collinearity_params: VIFCollinearityParams = Field(
        default_factory=VIFCollinearityParams
    )
    kmeans_collinearity_params: KMeansCollinearityParams = Field(
        default_factory=KMeansCollinearityParams
    )
    mi_collinearity_params: MICollinearityParams = Field(
        default_factory=MICollinearityParams
    )
    lasso_collinearity_params: LassoCollinearityParams = Field(
        default_factory=LassoCollinearityParams
    )
    hierarchical_clustering_collinearity_params: HierarchicalClusteringCollinearityParams = Field(
        default_factory=HierarchicalClusteringCollinearityParams
    )

    def __getitem__(self, key: CollinearityMethodEnum) -> BaseCollinearityParams:
        """
        Retrieve collinearity parameters based on the specified collinearity removal method.

        Parameters
        ----------
        key : CollinearityMethodEnum
            The collinearity removal method for which parameters are requested.
            Must be an instance of `CollinearityMethodEnum`.

        Returns
        -------
        Any
            The collinearity parameters corresponding to the specified method.
            This could be `correlation_collinearity_params`, `vif_collinearity_params`,
            or `kmeans_collinearity_params`.

        Raises
        ------
        ValueError
            If the provided key is not a valid type of collinearity removal method
            from `CollinearityMethodEnum`.
        """
        match key:
            case CollinearityMethodEnum.CORRELATION:
                return self.correlation_collinearity_params
            case CollinearityMethodEnum.VIF:
                return self.vif_collinearity_params
            case CollinearityMethodEnum.KMEANS:
                return self.kmeans_collinearity_params
            case CollinearityMethodEnum.MUTUAL_INFORMATION:
                return self.mi_collinearity_params
            case CollinearityMethodEnum.LASSO:
                return self.lasso_collinearity_params
            case CollinearityMethodEnum.HIERARCHICAL_CLUSTERING:
                return self.hierarchical_clustering_collinearity_params
            case _:
                raise ValueError(
                    'Given Key is not a type of collinearity removal method. Must be of type CollinearityMethodEnum.'
                )

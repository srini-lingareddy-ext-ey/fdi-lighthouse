from pydantic import Field

from lh_v2.util import BaseParamsModel

from .collinearity_params import CollinearityParams
from .full_pca_params import FullPCAParams
from .lag_params import LagParams
from .ranking_params import RankingParams


class DriverAnalysisParams(BaseParamsModel):
    """
    Parameters for driver analysis configuration.

    This class defines the parameters used to configure the driver analysis process,
    including PCA settings, lag analysis, ranking, and collinearity handling.

    Attributes
    ----------
    n_final_drivers_per_classification : int, default=3
        The number of final drivers to select for each classification category.
    use_full_pca : bool, default=False
        Whether to use full PCA (Principal Component Analysis) in the analysis.
    full_pca_params : FullPCAParams
        Parameters for configuring full PCA analysis. Initialized with default
        FullPCAParams if not provided.
    lag_params : LagParams
        Parameters for configuring lag analysis. Initialized with default
        LagParams if not provided.
    ranking_params : RankingParams
        Parameters for configuring driver ranking. Initialized with default
        RankingParams if not provided.
    collinearity_params : CollinearityParams
        Parameters for handling collinearity between variables. Initialized with
        default CollinearityParams if not provided.
    """

    n_final_drivers_per_classification: int = 3
    use_full_pca: bool = False
    full_pca_params: FullPCAParams = Field(default_factory=FullPCAParams)
    lag_params: LagParams = Field(default_factory=LagParams)
    ranking_params: RankingParams = Field(default_factory=RankingParams)
    collinearity_params: CollinearityParams = Field(default_factory=CollinearityParams)

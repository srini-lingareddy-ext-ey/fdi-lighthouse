from typing import Any

from pydantic import Field

from lh_v2.util import BaseParamsModel

from .ranking_params import RankingMethodsParams


class LagParams(BaseParamsModel):
    """
    Parameters for lag analysis in driver analysis.

    Parameters
    ----------
    b_lag : bool, default=True
        Whether to perform lag analysis.
    n_max_lag : int, default=12
        Maximum number of lag periods to consider.
    n_top_considered : int, default=4
        Number of top lagged variables to consider.
    ranking_methods : RankingMethodsParams
        Ranking methods configuration for lag analysis.
        Defaults to lagging-specific defaults.

    Methods
    -------
    __post_init__()
        Ensures n_max_lag is set to 0 when b_lag is False.
    """

    b_lag: bool = True
    n_max_lag: int = 12
    n_top_considered: int = 4
    ranking_methods: RankingMethodsParams = Field(
        default_factory=RankingMethodsParams.lagging_defaults
    )

    def model_post_init(self, _: Any) -> None:
        """
        Post-initialization method for the dataclass.

        This method is automatically called after the dataclass is initialized.
        It ensures that if `b_lag` is set to `False`, the `n_max_lag` attribute
        is set to 0.

        Returns
        -------
        None
            This method does not return any value.
        """
        if not self.b_lag:
            self.n_max_lag = 0
        return

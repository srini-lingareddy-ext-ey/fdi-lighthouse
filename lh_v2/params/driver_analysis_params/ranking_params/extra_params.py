from typing import Any, Sequence

from lh_v2.datatypes.driver_analysis_types.ranking_types import (
    DriverRankingEnum,
    DriverRankingLLMMetric,
)
from lh_v2.util import BaseParamsModel


class RankingMethodsParams(BaseParamsModel):
    """
    Parameters for configuring which driver ranking methods to include or exclude.

    Attributes
    ----------
    b_remove : bool
        Indicates whether methods are to be removed.
    methods_removed : Sequence[DriverRankingEnum]
        A sequence of driver ranking methods to be removed.
    b_selected : bool
        Indicates whether methods are to be selected.
    methods_selected : Sequence[DriverRankingEnum]
        A sequence of driver ranking methods to be selected.
    """

    b_remove: bool = True
    methods_removed: Sequence[DriverRankingEnum] = (
        DriverRankingEnum.RFE,
        DriverRankingEnum.BORUTA,
    )
    b_selected: bool = False
    methods_selected: Sequence[DriverRankingEnum] = ()

    @staticmethod
    def lagging_defaults() -> RankingMethodsParams:
        """
        Get RankingMethodsParams configured for lagging methods only.

        Returns
        -------
        RankingMethodsParams
            An instance of RankingMethodsParams with lagging methods selected.
        """
        return RankingMethodsParams(
            b_remove=False,
            methods_removed=(),
            b_selected=True,
            methods_selected=[
                DriverRankingEnum.SPEARMAN_CORRELATION,
                DriverRankingEnum.SLR,
                DriverRankingEnum.RIDGE,
            ],
        )


class LLMRankingParams(BaseParamsModel):
    """
    Parameters for configuring LLM-based driver ranking.

    Attributes
    ----------
    llm_metrics : Sequence[DriverRankingEnum]
        A sequence of driver ranking metrics to be used in the LLM ranking process.
    """

    llm_metrics: Sequence[DriverRankingLLMMetric] = (
        DriverRankingLLMMetric.FINAL_RANK,
        DriverRankingLLMMetric.AVG_RANK,
        DriverRankingLLMMetric.PEARSON_CORRELATION,
    )

    def model_post_init(self, _: Any):
        if DriverRankingLLMMetric.FINAL_RANK not in self.llm_metrics:
            self.llm_metrics = tuple(
                [DriverRankingLLMMetric.FINAL_RANK] + list(self.llm_metrics)
            )

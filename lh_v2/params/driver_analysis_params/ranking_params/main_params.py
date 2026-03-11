from typing import Any, Sequence

from pydantic import Field

from lh_v2.datatypes.driver_analysis_types.ranking_types import (
    DriverRankingEnum,
    DriverRankingMetric,
)
from lh_v2.util import BaseParamsModel

from .extra_params import RankingMethodsParams
from .method_params import (
    BaseRankingParams,
    BorutaRankingParams,
    LassoRankingParams,
    MARankingParams,
    MRMRRankingParams,
    PCARankingParams,
    PearsonRankingParams,
    RandomForestLightGBMRankingParams,
    RFELinearRegressionRankingParams,
    RFERankingParams,
    RidgeRankingParams,
    SLRRankingParams,
    SpearmanRankingParams,
    XGBoostRankingParams,
)


class RankingParams(BaseParamsModel):
    """
    Container for all ranking method parameters and configuration.

    This class provides a unified interface for accessing parameters for various
    driver ranking methods. It supports indexing by DriverRankingEnum to retrieve
    method-specific parameter objects.

    Attributes
    ----------
    methods : RankingMethodsParams
        Configuration for which ranking methods to include or exclude.
    pearson_ranking_params : PearsonRankingParams
        Parameters for Pearson correlation ranking.
    spearman_ranking_params : SpearmanRankingParams
        Parameters for Spearman correlation ranking.
    slr_ranking_params : SLRRankingParams
        Parameters for Simple Linear Regression ranking.
    ma_ranking_params : MARankingParams
        Parameters for Moving Average ranking.
    pca_ranking_params : PCARankingParams
        Parameters for PCA-based ranking.
    rfe_ranking_params : RFERankingParams
        Parameters for Recursive Feature Elimination (RFE) ranking.
    rfe_linear_regression_ranking_params : RFELinearRegressionRankingParams
        Parameters for Recursive Feature Elimination with Linear Regression ranking.
    lasso_ranking_params : LassoRankingParams
        Parameters for Lasso regression ranking.
    ridge_ranking_params : RidgeRankingParams
        Parameters for Ridge regression ranking.
    xgboost_ranking_params : XGBoostRankingParams
        Parameters for XGBoost ranking.
    boruta_ranking_params : BorutaRankingParams
        Parameters for Boruta feature selection ranking.
    random_forest_lightgbm_ranking_params : RandomForestLightGBMRankingParams
        Parameters for Random Forest LightGBM ranking.
    mrmr_ranking_params : MRMRRankingParams
        Parameters for mRMR ranking.

    Examples
    --------
    >>> params = RankingParams()
    >>> lasso_params = params[DriverRankingEnum.LASSO]
    >>> lasso_params.alpha
    0.1
    """

    methods: RankingMethodsParams = Field(default_factory=RankingMethodsParams)
    ranking_metrics: Sequence[DriverRankingMetric] = (
        DriverRankingMetric.FINAL_RANK,
        DriverRankingMetric.AVG_RANK,
        DriverRankingMetric.PEARSON_CORRELATION,
    )
    pearson_ranking_params: PearsonRankingParams = Field(
        default_factory=PearsonRankingParams
    )
    spearman_ranking_params: SpearmanRankingParams = Field(
        default_factory=SpearmanRankingParams
    )
    slr_ranking_params: SLRRankingParams = Field(default_factory=SLRRankingParams)
    ma_ranking_params: MARankingParams = Field(default_factory=MARankingParams)
    pca_ranking_params: PCARankingParams = Field(default_factory=PCARankingParams)
    rfe_ranking_params: RFERankingParams = Field(default_factory=RFERankingParams)
    rfe_linear_regression_ranking_params: RFELinearRegressionRankingParams = Field(
        default_factory=RFELinearRegressionRankingParams
    )
    lasso_ranking_params: LassoRankingParams = Field(default_factory=LassoRankingParams)
    ridge_ranking_params: RidgeRankingParams = Field(default_factory=RidgeRankingParams)
    xgboost_ranking_params: XGBoostRankingParams = Field(
        default_factory=XGBoostRankingParams
    )
    boruta_ranking_params: BorutaRankingParams = Field(
        default_factory=BorutaRankingParams
    )
    random_forest_lightgbm_ranking_params: RandomForestLightGBMRankingParams = Field(
        default_factory=RandomForestLightGBMRankingParams
    )
    mrmr_ranking_params: MRMRRankingParams = Field(default_factory=MRMRRankingParams)

    def model_post_init(self, _: Any) -> None:
        if DriverRankingMetric.FINAL_RANK not in self.ranking_metrics:
            self.ranking_metrics = tuple(
                [DriverRankingMetric.FINAL_RANK] + list(self.ranking_metrics)
            )
        return

    def __getitem__(self, key: DriverRankingEnum) -> BaseRankingParams:
        """
        Retrieve the ranking parameters for a specific driver ranking method.

        Parameters
        ----------
        key : DriverRankingEnum
            The driver ranking method for which parameters are to be retrieved.

        Returns
        -------
        BaseRankingParams
            The ranking parameters corresponding to the specified driver ranking method.

        Raises
        ------
        ValueError
            If the provided key is not a valid DriverRankingEnum value.
        """
        match key:
            case DriverRankingEnum.PEARSON_CORRELATION:
                return self.pearson_ranking_params
            case DriverRankingEnum.SPEARMAN_CORRELATION:
                return self.spearman_ranking_params
            case DriverRankingEnum.SLR:
                return self.slr_ranking_params
            case DriverRankingEnum.MOVING_AVERAGE:
                return self.ma_ranking_params
            case DriverRankingEnum.PCA:
                return self.pca_ranking_params
            case DriverRankingEnum.RFE:
                return self.rfe_ranking_params
            case DriverRankingEnum.RFE_LINEAR_REGRESSION:
                return self.rfe_linear_regression_ranking_params
            case DriverRankingEnum.LASSO:
                return self.lasso_ranking_params
            case DriverRankingEnum.RIDGE:
                return self.ridge_ranking_params
            case DriverRankingEnum.XGBOOST:
                return self.xgboost_ranking_params
            case DriverRankingEnum.BORUTA:
                return self.boruta_ranking_params
            case DriverRankingEnum.RANDOM_FOREST_LIGHTGBM:
                return self.random_forest_lightgbm_ranking_params
            case DriverRankingEnum.MRMR:
                return self.mrmr_ranking_params
            case _:
                raise ValueError(
                    'Given Key is not a type of driver ranking method. Must be of type DriverRankingEnum.'
                )

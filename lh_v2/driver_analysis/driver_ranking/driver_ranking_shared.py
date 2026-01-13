import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params as params

from .driver_ranking_methods import (
    AbstractRankingMethod,
    BorutaRanking,
    LassoRanking,
    MovingAverageRanking,
    MRMRRanking,
    PCARanking,
    PearsonCorrelationRanking,
    RandomForestLightGBMRanking,
    RFELinearRegressionRanking,
    RFERanking,
    RidgeRanking,
    SLRRanking,
    SpearmanCorrelationRanking,
    XGBoostRanking,
)

RANKING_METHOD_MAP: dict[rdt.DriverRankingEnum, type[AbstractRankingMethod]] = {
    rdt.DriverRankingEnum.PEARSON_CORRELATION: PearsonCorrelationRanking,
    rdt.DriverRankingEnum.SPEARMAN_CORRELATION: SpearmanCorrelationRanking,
    rdt.DriverRankingEnum.SLR: SLRRanking,
    rdt.DriverRankingEnum.MOVING_AVERAGE: MovingAverageRanking,
    rdt.DriverRankingEnum.PCA: PCARanking,
    rdt.DriverRankingEnum.RFE: RFERanking,
    rdt.DriverRankingEnum.RFE_LINEAR_REGRESSION: RFELinearRegressionRanking,
    rdt.DriverRankingEnum.RANDOM_FOREST_LIGHTGBM: RandomForestLightGBMRanking,
    rdt.DriverRankingEnum.LASSO: LassoRanking,
    rdt.DriverRankingEnum.RIDGE: RidgeRanking,
    rdt.DriverRankingEnum.XGBOOST: XGBoostRanking,
    rdt.DriverRankingEnum.BORUTA: BorutaRanking,
    rdt.DriverRankingEnum.MRMR: MRMRRanking,
}


def select_methods(
    methods_params: params.RankingMethodsParams,
) -> list[rdt.DriverRankingEnum]:
    """
    Configure which driver ranking methods to use based on configuration parameters.

    This method determines the ranking methods to be used for driver analysis based on
    the configuration in ranking_params. It supports three modes:
    1. Using all available ranking methods (default)
    2. Using all methods except those specified for removal
    3. Using only explicitly selected methods

    Returns
    -------
    list[rdt.DriverRankingEnum]
        List of Ranking methods to be used.

    Raises
    ------
    ValueError
        If both b_remove and b_selected are set to True, which is an invalid configuration

    Notes
    -----
    Exactly one of the following must be true:
    - Neither b_remove nor b_selected is True (use all methods)
    - b_remove is True (use all except specified)
    - b_selected is True (use only specified)
    """
    # Check for invalid configuration: both flags cannot be True simultaneously
    if methods_params.b_remove and methods_params.b_selected:
        raise ValueError(
            'b_remove and b_selected cannot both be true. Must select one or neither to be True.'
        )

    # Default case: use all available ranking methods if no filtering is specified
    if not methods_params.b_remove and not methods_params.b_selected:
        return [k for k in rdt.DriverRankingEnum]

    # Case: exclude specific methods from the full set
    if methods_params.b_remove:
        base_list = [
            k for k in rdt.DriverRankingEnum
        ]  # Start with all available methods
        for method in methods_params.methods_removed:
            base_list.remove(method)  # Remove each specified method
        return base_list

    # Case: include only specifically selected methods
    if methods_params.b_selected:
        return list(methods_params.methods_selected)

    raise ValueError(
        'Ranking General Params not configured properly, '
        'this should not happen, if it does talk to a developer.'
    )

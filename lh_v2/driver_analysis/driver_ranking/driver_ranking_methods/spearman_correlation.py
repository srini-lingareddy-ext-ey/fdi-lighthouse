import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod


class SpearmanCorrelationRanking(AbstractRankingMethod):
    """
    Ranks drivers using Spearman correlation coefficients.

    This class implements driver ranking by calculating the absolute Spearman
    correlation coefficients between account data and each driver.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Object containing account and driver data to be analyzed.
    method_params : dr_params.SpearmanRankingParams
        Parameters specific to Spearman correlation ranking method.

    Methods
    -------
    apply()
        Calculate absolute Spearman correlation coefficients for each driver.
    name()
        Return the name of the ranking method.

    Notes
    -----
    The Spearman correlation measures the strength and direction of monotonic
    relationship between two variables. This implementation takes the absolute
    values of the correlations for ranking purposes.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.SpearmanRankingParams,
    ):
        assert isinstance(method_params, dr_params.SpearmanRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.SpearmanRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name() -> str:
        return 'SpearmanCorrelation'

    def apply(self) -> ArrayF:
        """
        Compute the absolute Spearman correlation coefficients between account data
        and each driver in the drivers array.

        Returns
        -------
        np.ndarray
            A 1D array of absolute Spearman correlation coefficients, with the same
            dtype as `self.info.np_dtype`.
        """
        corrs = np.zeros((self.info.drivers.arr.shape[0],), dtype=self.info.np_dtype)
        for k in range(self.info.drivers.arr.shape[0]):
            corrs[k] = stats.spearman_correlation(
                self.info.account.arr, self.info.drivers[k]
            )
        return np.abs(corrs).astype(self.info.np_dtype)

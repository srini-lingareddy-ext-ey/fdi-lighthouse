import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingEnum
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod


class PearsonCorrelationRanking(AbstractRankingMethod):
    """
    A ranking method that uses absolute Pearson correlation coefficients to rank drivers.
    This class implements a ranking strategy based on the absolute value of the Pearson
    correlation coefficient between account data and each driver. Higher correlation values
    (in absolute terms) indicate stronger relationships between drivers and the account data.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Object containing account and driver data to analyze.
    method_params : dr_params.PearsonRankingParams
        Parameters specific to the Pearson correlation ranking method.
        Default is an instance of PearsonRankingParams with default values.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored reference to the account and driver data.
    method_params : dr_params.PearsonRankingParams
        Parameters for the ranking method.

    Methods
    -------
    apply()
        Computes absolute Pearson correlation coefficients between account data and each driver.
    name()
        Returns the name of the ranking method.

    See Also
    --------
    AbstractRankingMethod : Parent class that defines the interface for ranking methods.
    The Pearson correlation coefficient measures the linear correlation between two variables.
    Absolute values are used to rank drivers, as both strong positive and negative correlations
    are considered important.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.PearsonRankingParams,
    ):
        assert isinstance(method_params, dr_params.PearsonRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.PearsonRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name() -> str:
        return 'PearsonCorrelation'

    @staticmethod
    def get_enum() -> DriverRankingEnum:
        return DriverRankingEnum.PEARSON_CORRELATION

    def apply(self) -> ArrayF:
        """
        Compute the absolute Pearson correlation coefficients between account data
        and each driver in the drivers array.

        Returns
        -------
        np.ndarray
            A 1D array of absolute Pearson correlation coefficients, with the same
            dtype as `self.info.np_dtype`.

        Notes
        -----
        - The method iterates over each driver in the `self.info.drivers.arr` array
          and computes the Pearson correlation coefficient with `self.info.account.arr`.
        - The resulting correlation coefficients are converted to their absolute
          values and cast to the specified numpy dtype (`self.info.np_dtype`).
        """
        corrs = np.zeros((self.info.drivers.arr.shape[0],), dtype=self.info.np_dtype)
        for k in range(self.info.drivers.arr.shape[0]):
            corrs[k] = stats.pearson_correlation(
                self.info.account.arr, self.info.drivers[k]
            )
        return np.abs(corrs).astype(self.info.np_dtype)

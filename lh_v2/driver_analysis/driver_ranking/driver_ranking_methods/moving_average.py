import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod


class MovingAverageRanking(AbstractRankingMethod):
    """
    A ranking method that evaluates drivers based on the correlation between
    their moving averages and account data.

    This class computes moving averages for each driver time series and then
    calculates Pearson correlation coefficients between these moving averages
    and the account data. Higher correlation indicates a stronger relationship
    between a driver and the account.

    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Container for account and driver data.
    method_params : dr_params.MARankingParams
        Parameters for the moving average ranking method, including:
        - ma_type: Type of moving average to compute
        - tail_len: Length of the moving average window
        - a: Parameter used in certain moving average calculations

    Attributes
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.MARankingParams
        Stored method parameters.

    Methods
    name()
        Returns the name identifier for this ranking method.
    apply()
        Computes correlation coefficients between driver moving averages and account data.

    See Also
    --------
    AbstractRankingMethod : Parent class defining the ranking method interface.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.MARankingParams,
    ):
        assert isinstance(method_params, dr_params.MARankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.MARankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name() -> str:
        return 'MovingAverage'

    def apply(self) -> ArrayF:
        """
        Compute the correlation between the moving average of each driver and the account data.

        Parameters
        ----------
        a : float, optional
            A parameter used in the moving average computation, by default 1/2.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the correlation coefficients between the moving
            average of each driver and the account data.
        """
        # Calculate moving averages for all driver data using parameters from method_params
        ma_drivers = stats.moving_average(
            self.info.drivers.arr,
            self.method_params.ma_type,
            self.method_params.tail_len,
            self.method_params.a,
        ).astype(self.info.np_dtype)

        # Initialize array to store correlation coefficients
        corrs = np.zeros((ma_drivers.shape[0],), self.info.np_dtype)

        # Calculate Pearson correlation between each driver's moving average and account data
        for k in range(ma_drivers.shape[0]):
            corrs[k] = stats.pearson_correlation(ma_drivers[k], self.info.account.arr)

        # Return the array of correlation coefficients
        return corrs.astype(self.info.np_dtype)

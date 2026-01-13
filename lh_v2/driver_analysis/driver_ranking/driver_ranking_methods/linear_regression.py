import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod, error_norm


class SLRRanking(AbstractRankingMethod):
    """
    Simple Linear Regression (SLR) based driver ranking method.

    This class implements a driver ranking approach based on Simple Linear Regression.
    It evaluates drivers by calculating how well each driver's data can predict
    the account data using a simple linear model (y = ax + b). Lower error values
    indicate better rankings.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Object containing account and driver data for analysis.
    method_params : dr_params.SLRRankingParams
        Parameters specific to the SLR ranking method.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.SLRRankingParams
        Stored parameters for the SLR ranking method.

    Methods
    -------
    apply()
        Computes the ranking errors for each driver using SLR.
    name()
        Returns the name identifier for this ranking method.

    See Also
    --------
    AbstractRankingMethod : Parent class defining the ranking interface.

    The ranking is determined by how well each driver's data linearly predicts
    the account data, with lower prediction errors resulting in higher rankings.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.SLRRankingParams,
    ):
        assert isinstance(method_params, dr_params.SLRRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.SLRRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        super().__init__(account_driver_info, method_params)
        return

    @staticmethod
    def name() -> str:
        return 'SLR'

    def apply(self) -> ArrayF:
        """
        Applies the Simple Linear Regression (SLR) ranking method to the drivers.

        This method computes the SLR parameters (slope and intercept) for each driver
        with respect to the account data. It then calculates the error between the
        predicted and actual account data using the given norm function.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the error values for each driver. Smaller
            errors indicate better rankings.

        Notes
        -----
        - The method iterates over each driver in the `self.info.drivers.arr` array.
        - For each driver, it computes the SLR parameters (slope and intercept) using
          the `stats.slr` function.
        - The error is calculated as the norm of the difference between the actual
          account data and the predicted account data obtained by applying the SLR
          parameters to the driver data.
        """
        # Initialize arrays to store slopes, intercepts, and error values for each driver
        a = np.zeros((self.info.drivers.arr.shape[0],), self.info.np_dtype)
        b = np.zeros((self.info.drivers.arr.shape[0],), self.info.np_dtype)
        error = np.zeros((self.info.drivers.arr.shape[0],), self.info.np_dtype)

        # Process each driver individually
        for k in range(self.info.drivers.arr.shape[0]):
            # Calculate the SLR parameters (slope and intercept) for current driver
            ph = stats.slr(self.info.drivers[k], self.info.account.arr)
            a[k] = ph[0]  # Store the slope
            b[k] = ph[1]  # Store the intercept

            # Compute the prediction error:
            # 1. Apply SLR parameters to predict account data from driver data
            # 2. Subtract prediction from actual account data to get error
            # 3. Apply norm function to get a scalar error value
            error[k] = error_norm(
                self.info.account.arr
                - stats.apply_slr(float(a[k]), float(b[k]), self.info.drivers[k])
            )

        # Return the array of errors - smaller errors indicate better predictors
        return -error

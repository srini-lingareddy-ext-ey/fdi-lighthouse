import warnings

import numpy as np
import sklearn.linear_model as sklm
import sklearn.model_selection as skm

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.datatypes.driver_analysis_types.ranking_types import DriverRankingEnum
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)  # ignore convergence warnin


class RidgeRanking(AbstractRankingMethod):
    """
    A class that implements driver ranking using Ridge Regression.

    This class ranks drivers based on their importance determined by the Ridge Regression
    coefficients. The ranking is computed by performing multiple iterations of Ridge
    Regression, each time splitting the data into training and testing sets, and then
    averaging the absolute coefficient values across iterations.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Container for account and driver data.
    params : dr_params.RidgeRankingParams
        Parameters for the Ridge Regression ranking method.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.RidgeRankingParams
        Stored parameters for the ranking method.

    - The method performs multiple iterations, each with a different random split of data.
    - Driver importance is determined by averaging absolute coefficient values across iterations.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.RidgeRankingParams,
    ):
        assert isinstance(method_params, dr_params.RidgeRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.RidgeRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name():
        return 'Ridge'

    @staticmethod
    def get_enum() -> DriverRankingEnum:
        return DriverRankingEnum.RIDGE

    def apply(self) -> ArrayF:
        """
        Applies the Ridge Regression ranking method to rank drivers.

        This method uses Ridge Regression to determine the importance of each driver
        based on the Ridge coefficients. It performs multiple iterations of Ridge
        Regression, each time splitting the data into training and testing sets.
        The average absolute coefficient values across iterations are used as the
        ranking score.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the average absolute coefficient values for each driver
            across multiple iterations of Ridge Regression. Higher values indicate
            higher importance.

        Notes
        -----
        - The driver data is normalized before applying Ridge Regression.
        - The method performs `n_iter` iterations of Ridge Regression.
        - In each iteration, the data is split into training and testing sets using
          `sklearn.model_selection.train_test_split`.
        - A Ridge Regression model is fitted to the training data, and the absolute
          coefficient values are used to determine driver importance.
        - The average absolute coefficient value of each driver is calculated across all iterations.
        """
        # Normalize driver data for more stable Ridge regression results
        drivers_norm = stats.normalize_arr(self.info.drivers.arr).astype(
            self.info.np_dtype
        )

        # Initialize array to store coefficient values from each iteration
        arr_ridge = np.zeros(
            (self.method_params.n_iter, self.info.drivers.arr.shape[0]),
            dtype=self.info.np_dtype,
        )

        # Run multiple iterations of Ridge regression with different random states
        for k in range(self.method_params.n_iter):
            # Split data into training and testing sets (80/20 split)
            # Transpose drivers_norm to get features as columns (sklearn expected format)
            X_train, X_test, y_train, y_test = skm.train_test_split(  # pyright: ignore[reportUnusedVariable]
                drivers_norm.T,
                self.info.account.arr,
                test_size=0.2,
                random_state=k,  # Different random state each iteration for robust averaging
            )

            # Create and fit Ridge regression model with regularization parameter from config
            ridge = sklm.Ridge(alpha=self.method_params.alpha, random_state=k)
            ridge.fit(X_train, y_train)

            # Store absolute coefficient values for this iteration
            # Absolute values ensure we capture both positive and negative importance
            arr_ridge[k] = np.abs(ridge.coef_)

        # Average absolute coefficient values across iterations to get final ranking scores
        # Higher values indicate more important drivers
        return arr_ridge.mean(axis=0).astype(self.info.np_dtype)

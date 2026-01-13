import warnings

import numpy as np
import sklearn.linear_model as sklm
import sklearn.model_selection as skm

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)  # ignore convergence warnings


class LassoRanking(AbstractRankingMethod):
    """
    A class that implements driver ranking using Lasso Regression.

    This class ranks drivers based on their importance determined by the Lasso Regression
    coefficients. The ranking is computed by performing multiple iterations of Lasso
    Regression, each time splitting the data into training and testing sets, and then
    counting how frequently each driver is selected across iterations.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Container for account and driver data.
    params : dr_params.LassoRankingParams
        Parameters for the Lasso Regression ranking method.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.LassoRankingParams
        Stored parameters for the ranking method.

    - The method performs multiple iterations, each with a different random split of data.
    - Driver importance is determined by how frequently their coefficients exceed a threshold.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.LassoRankingParams,
    ):
        assert isinstance(method_params, dr_params.LassoRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.LassoRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name() -> str:
        return 'Lasso'

    def apply(self) -> ArrayF:
        """
        Applies the Lasso Regression ranking method to rank drivers.

        This method uses Lasso Regression to determine the importance of each driver
        based on the Lasso coefficients. It performs multiple iterations of Lasso
        Regression, each time splitting the data into training and testing sets.
        The frequency of selection of each driver across iterations is used as the
        ranking score.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the frequency of selection for each driver
            across multiple iterations of Lasso Regression. Higher values indicate
            higher importance.

        Notes
        -----
        - The driver data is normalized before applying Lasso Regression.
        - The method performs `n_iter` iterations of Lasso Regression.
        - In each iteration, the data is split into training and testing sets using
          `sklearn.model_selection.train_test_split`.
        - A Lasso Regression model is fitted to the training data, and the coefficients
          are used to determine the selected features.
        - The frequency of selection of each driver is calculated by summing the
          selection counts across all iterations.
        """
        # Normalize driver data for more stable Lasso regression results
        drivers_norm = stats.normalize_arr(self.info.drivers.arr).astype(
            self.info.np_dtype
        )

        # Initialize array to track which drivers are selected in each iteration
        arr_lasso = np.zeros(
            (self.method_params.n_iter, self.info.drivers.arr.shape[0]), dtype=np.int32
        )

        # Run multiple iterations of Lasso regression with different random states
        for k in range(self.method_params.n_iter):
            # Split data into training and testing sets (80/20 split)
            # Transpose drivers_norm to get features as columns (sklearn expected format)
            X_train, _, y_train, _ = skm.train_test_split(
                drivers_norm.T,
                self.info.account.arr,
                test_size=0.2,
                random_state=k,  # Using different split for each iteration
            )

            # Create and fit Lasso regression model with regularization parameter alpha=0.1
            lasso = sklm.Lasso(
                alpha=self.method_params.alpha,
                random_state=2 * k,
                max_iter=self.method_params.max_iter,
            )  # Different random state each iteration
            lasso.fit(X_train, y_train)

            # Consider a feature selected if its coefficient is above threshold (0.1)
            selected_features = lasso.coef_ > 0.1

            # Record which drivers were selected in this iteration (1=selected, 0=not selected)
            arr_lasso[k] = selected_features.astype(int)

        # Sum across iterations to get selection frequency for each driver
        # Higher values indicate more important drivers
        return arr_lasso.sum(axis=0).astype(self.info.np_dtype)

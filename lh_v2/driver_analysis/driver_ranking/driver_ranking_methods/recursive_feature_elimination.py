import warnings

import numpy as np
import sklearn.ensemble as ske
import sklearn.feature_selection as skf
import sklearn.model_selection as skm

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)  # ignore convergence warning


class RFERanking(AbstractRankingMethod):
    """
    Ranking method that uses Recursive Feature Elimination (RFE) to rank drivers.

    This class implements a driver ranking approach based on Recursive Feature Elimination (RFE)
    with a Random Forest Regressor as the base estimator. It performs multiple iterations,
    each with a different train-test split, and averages the rankings across iterations
    to provide more stable results.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Object containing information about the account and drivers.
    method_params : dr_params.RFERankingParams
        Parameters for the RFE ranking method, including number of iterations and
        number of estimators for the Random Forest.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.RFERankingParams
        Stored parameters for the ranking method.

    Methods
    -------
    apply()
        Applies the RFE ranking method and returns driver rankings.
    name()
        Returns the name identifier for this ranking method.

    The implementation averages rankings from multiple RFE iterations with different
    random seeds to provide more stable and reliable driver importance rankings.
    Higher values in the returned rankings indicate more important drivers.
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.RFERankingParams,
    ):
        assert isinstance(method_params, dr_params.RFERankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.RFERankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name() -> str:
        return 'RFE'

    def apply(self) -> ArrayF:
        """
        Applies the Recursive Feature Elimination (RFE) ranking method to rank drivers.

        This method uses a Random Forest Regressor as the estimator for RFE. It performs
        multiple iterations of RFE, each time splitting the data into training and testing
        sets. The rankings from all iterations are averaged to produce the final ranking.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the averaged inverse rankings for each driver.
            Higher values indicate better rankings.

        Notes
        -----
        - The method performs `n_iter` iterations of RFE.
        - In each iteration, the data is split into training and testing sets using
          `sklearn.model_selection.train_test_split`.
        - A Random Forest Regressor with `n_estimators` trees is used as the estimator
          for RFE.
        - The rankings are averaged across all iterations, and the inverse of the
          averaged rankings is returned.
        """
        # Initialize array to store rankings from each iteration
        rankings = np.zeros(
            (self.method_params.n_iter, self.info.drivers.arr.shape[0]),
            self.info.np_dtype,
        )

        # Run multiple iterations of RFE with different train-test splits
        for k in range(self.method_params.n_iter):
            # Split data into training and testing sets
            # Note: Transpose drivers.arr to get features as columns
            X_train, X_test, y_train, y_test = skm.train_test_split(  # pyright: ignore[reportUnusedVariable]
                self.info.drivers.arr.T,  # Transpose to get features as columns
                self.info.account.arr,  # Account data as target variable
                test_size=0.1,  # Use 10% of data for testing
                random_state=k,  # Different seed each iteration for stability
            )

            # Create Random Forest Regressor as the estimator for RFE
            model = ske.RandomForestRegressor(
                n_estimators=self.method_params.n_estimators,  # Number of trees
                random_state=k,  # Same seed for reproducibility
            )

            # Create RFE object that will rank features by recursively eliminating them
            rfe = skf.RFE(estimator=model, n_features_to_select=1)

            # Fit RFE to training data
            rfe.fit(X_train, y_train)

            # Extract feature support mask (True for selected features)
            # selected_features = rfe.support_.astype(int)

            # Get feature rankings (smaller number = higher importance)
            feature_ranking = rfe.ranking_

            # Store this iteration's rankings
            rankings[k] = feature_ranking

        # Average the rankings across all iterations
        # Convert to scores: lower rankings (more important) become higher scores
        # Add 1 to avoid division by zero, then take reciprocal
        return (1 / (rankings.mean(axis=0) + 1)).astype(self.info.np_dtype)

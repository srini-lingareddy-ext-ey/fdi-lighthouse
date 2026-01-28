import warnings

import numpy as np
from sklearn.feature_selection import mutual_info_regression

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)


class MRMRRanking(AbstractRankingMethod):
    """
        A class that implements driver ranking using mRMR (minimum Redundancy Maximum Relevance).
    whe
        mRMR is a feature selection algorithm that selects features by balancing:
        - Maximum Relevance: Features highly correlated with the target
        - Minimum Redundancy: Features that are minimally correlated with each other

        The algorithm uses mutual information to measure both relevance and redundancy,
        making it capable of capturing non-linear relationships.

        Parameters
        ----------
        account_driver_info : dts.account_driver_types.AccountDriverGroup
            Container for account and driver data.
        params : dr_params.MRMRRankingParams
            Parameters for the mRMR ranking method.

        Attributes
        ----------
        info : dts.account_driver_types.AccountDriverGroup
            Stored account and driver information.
        method_params : dr_params.MRMRRankingParams
            Stored parameters for the ranking method.

        Notes
        -----
        - Uses greedy forward selection to build feature set
        - First feature: maximum mutual information with target
        - Subsequent features: maximize (relevance - average redundancy)
        - Handles both linear and non-linear relationships via mutual information
        - Automatically avoids multicollinearity by penalizing redundant features
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.MRMRRankingParams,
    ):
        assert isinstance(method_params, dr_params.MRMRRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.MRMRRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name():
        return 'mRMR'

    def _mutual_information(self, X: ArrayF, y: ArrayF) -> ArrayF:
        """
        Compute mutual information between each feature in X and target y.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (samples x features).
        y : np.ndarray
            Target variable.

        Returns
        -------
        np.ndarray
            Mutual information scores for each feature.
        """
        # mutual_info_regression expects samples x features
        mi_scores = mutual_info_regression(
            X, y, discrete_features='auto', random_state=42
        )
        return mi_scores.astype(self.info.np_dtype)

    def _mutual_information_pairwise(
        self, X: ArrayF, feature_idx: int, selected_indices: list[int]
    ) -> float:
        """
        Compute average mutual information between a feature and already-selected features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (samples x features).
        feature_idx : int
            Index of the feature to evaluate.
        selected_indices : list[int]
            Indices of already-selected features.

        Returns
        -------
        float
            Average mutual information with selected features (redundancy).
        """
        if len(selected_indices) == 0:
            return 0.0

        redundancies = []
        feature_data = X[:, feature_idx]

        for selected_idx in selected_indices:
            selected_feature_data = X[:, selected_idx]
            # Compute MI between two features
            # Reshape to 2D for sklearn
            mi = mutual_info_regression(
                selected_feature_data.reshape(-1, 1),
                feature_data,
                discrete_features='auto',
                random_state=42,
            )[0]
            redundancies.append(mi)

        return float(np.mean(redundancies))

    def apply(self) -> ArrayF:
        """
        Applies the mRMR algorithm to rank drivers.

        The algorithm:
        1. Computes relevance (MI with target) for all features
        2. Selects first feature with maximum relevance
        3. For each subsequent feature, computes:
           mRMR_score = Relevance - Average_Redundancy
        4. Selects feature with highest mRMR score
        5. Repeats until n_features selected

        Returns
        -------
        np.ndarray
            A 1D numpy array containing mRMR scores for each driver.
            Higher scores = selected earlier in the greedy process.
            Drivers selected get scores from n_features down to 1.
            Unselected drivers get score of 0.

        Notes
        -----
        - Uses mutual information to capture non-linear relationships
        - Greedy forward selection ensures diversity in selected features
        - Balances predictive power (relevance) with feature diversity (redundancy)
        - Unlike correlation methods, works with non-linear dependencies
        - Unlike LASSO, explicitly minimizes feature redundancy
        """
        # Normalize driver data
        drivers_norm = stats.normalize_arr(self.info.drivers.arr).astype(
            self.info.np_dtype
        )
        # Transpose to get samples x features format for sklearn
        X = drivers_norm.T
        y = self.info.account.arr
        n_features_total = X.shape[1]

        # Limit selection to available features
        n_to_select = min(self.method_params.n_features, n_features_total)

        # Compute relevance scores (MI with target) for all features
        relevance_scores = self._mutual_information(X, y)

        # Initialize tracking arrays
        selected_indices = []
        mrmr_scores = np.zeros(n_features_total, dtype=self.info.np_dtype)

        # Greedy feature selection
        for selection_step in range(n_to_select):
            best_score = -np.inf
            best_idx = -1

            # Evaluate each unselected feature
            for feature_idx in range(n_features_total):
                if feature_idx in selected_indices:
                    continue

                # Relevance term
                relevance = relevance_scores[feature_idx]

                # Redundancy term (average MI with already-selected features)
                if len(selected_indices) > 0:
                    redundancy = self._mutual_information_pairwise(
                        X, feature_idx, selected_indices
                    )
                else:
                    redundancy = 0.0

                # mRMR criterion: maximize relevance, minimize redundancy
                mrmr_criterion = relevance - redundancy

                if mrmr_criterion > best_score:
                    best_score = mrmr_criterion
                    best_idx = feature_idx

            # Select the best feature
            if best_idx >= 0:
                selected_indices.append(best_idx)
                # Assign score based on selection order (earlier = higher score)
                mrmr_scores[best_idx] = n_to_select - selection_step

        return mrmr_scores

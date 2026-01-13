import warnings

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestRegressor

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)


class BorutaRanking(AbstractRankingMethod):
    """
    A class that implements driver ranking using the Boruta feature selection algorithm.

    Boruta is an all-relevant feature selection method that identifies all features that are
    statistically relevant to the target variable. It works by comparing the importance of
    real features against shadow features (shuffled copies) using Random Forest and statistical
    hypothesis testing.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Container for account and driver data.
    params : dr_params.BorutaRankingParams
        Parameters for the Boruta ranking method.

    Attributes
    ----------
    info : dts.account_driver_types.AccountDriverGroup
        Stored account and driver information.
    method_params : dr_params.BorutaRankingParams
        Stored parameters for the ranking method.

    Notes
    -----
    - Creates shadow features by shuffling real features to establish a baseline
    - Uses Random Forest to compute feature importances
    - Performs statistical tests to classify features as confirmed, rejected, or tentative
    - Features are ranked based on their confirmation frequency across iterations
    - Unlike LASSO/Ridge, Boruta finds ALL relevant features, not just the top ones
    """

    def __init__(
        self,
        account_driver_info: dts.account_driver_types.AccountDriverGroup,
        method_params: dr_params.BorutaRankingParams,
    ):
        assert isinstance(method_params, dr_params.BorutaRankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.BorutaRankingParams}, Params were of type {type(method_params)}'
        )
        self.info = account_driver_info
        self.method_params = method_params
        return

    @staticmethod
    def name():
        return 'Boruta'

    def _create_shadow_features(self, X: ArrayF, random_state: int) -> ArrayF:
        """
        Create shadow features by shuffling original features.

        Parameters
        ----------
        X : np.ndarray
            Original feature matrix (samples x features).
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Shadow features with same shape as X but shuffled values.
        """
        rng = np.random.RandomState(random_state)
        shadow = X.copy()
        # Shuffle each feature independently
        for i in range(shadow.shape[1]):
            rng.shuffle(shadow[:, i])
        return shadow

    def _get_feature_importances(
        self, X: ArrayF, y: ArrayF, random_state: int
    ) -> ArrayF:
        """
        Compute feature importances using Random Forest.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix including both real and shadow features.
        y : np.ndarray
            Target variable (account data).
        random_state : int
            Random seed for Random Forest.

        Returns
        -------
        np.ndarray
            Feature importance scores for all features.
        """
        rf = RandomForestRegressor(
            n_estimators=self.method_params.n_estimators,
            max_depth=self.method_params.max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        return rf.feature_importances_

    def apply(self) -> ArrayF:
        """
        Applies the Boruta feature selection algorithm to rank drivers.

        The algorithm:
        1. Creates shadow features by shuffling real features
        2. Trains Random Forest on combined real + shadow features
        3. Compares real feature importances to max shadow importance
        4. Uses binomial test to determine if feature is significantly better
        5. Confirms/rejects features based on statistical significance
        6. Repeats for max_iter or until all features classified

        Returns
        -------
        np.ndarray
            A 1D numpy array containing confirmation scores for each driver.
            Higher values indicate features that were confirmed more often,
            meaning they are more relevant to the target.

        Notes
        -----
        - Confirmation score = number of times feature was statistically
          significant across all iterations
        - Features consistently better than shadow features get higher scores
        - Unlike correlation methods, Boruta captures non-linear relationships
        - Unlike LASSO, Boruta finds all relevant features, not just sparse subset
        """
        # Normalize driver data
        drivers_norm = stats.normalize_arr(self.info.drivers.arr).astype(
            self.info.np_dtype
        )
        # Transpose to get samples x features format for sklearn
        X = drivers_norm.T
        y = self.info.account.arr
        n_features = X.shape[1]

        # Track how many times each feature is confirmed as important
        confirmation_counts = np.zeros(n_features, dtype=np.float32)
        # Track total iterations each feature was tested
        test_counts = np.zeros(n_features, dtype=np.float32)
        # Track cumulative feature importances for tie-breaking
        importance_sums = np.zeros(n_features, dtype=np.float32)

        # Feature states: 0=tentative, 1=confirmed, -1=rejected
        feature_states = np.zeros(n_features, dtype=np.int8)

        for iteration in range(self.method_params.max_iter):
            # Get indices of features still being tested
            tentative_mask = feature_states == 0
            if not np.any(tentative_mask):
                # All features classified, stop early
                break

            # Create shadow features
            shadow_features = self._create_shadow_features(X, random_state=iteration)

            # Combine real and shadow features
            # Only include tentative real features
            X_tentative = X[:, tentative_mask]
            X_combined = np.hstack([X_tentative, shadow_features])

            # Get feature importances
            importances = self._get_feature_importances(
                X_combined, y, random_state=iteration
            )

            # Split importances into real and shadow
            n_tentative = X_tentative.shape[1]
            real_importances = importances[:n_tentative]
            shadow_importances = importances[n_tentative:]

            # Compute shadow threshold (percentile of shadow importances)
            if self.method_params.perc == 100:
                shadow_threshold = np.max(shadow_importances)
            else:
                shadow_threshold = np.percentile(
                    shadow_importances, self.method_params.perc
                )

            # Compare each tentative feature to shadow threshold
            tentative_indices = np.where(tentative_mask)[0]
            for i, feature_idx in enumerate(tentative_indices):
                test_counts[feature_idx] += 1
                # Track importance for tie-breaking
                importance_sums[feature_idx] += real_importances[i]

                # Feature is important if it beats shadow threshold
                if real_importances[i] > shadow_threshold:
                    confirmation_counts[feature_idx] += 1

                # After enough tests, make final decision
                if test_counts[feature_idx] >= 20:  # Need at least 20 tests
                    # Binomial test: is confirmation rate significantly > 50%?
                    p_value = binomtest(
                        int(confirmation_counts[feature_idx]),
                        int(test_counts[feature_idx]),
                        0.5,
                        alternative='greater',
                    ).pvalue

                    if p_value < self.method_params.alpha:
                        # Significantly better than random
                        feature_states[feature_idx] = 1  # Confirmed
                    elif test_counts[feature_idx] >= self.method_params.max_iter * 0.8:
                        # Not significant after many iterations
                        feature_states[feature_idx] = -1  # Rejected

        # Create ranking scores:
        # Primary: confirmation rate (higher = more confirmed)
        # Secondary: average importance (tie-breaker for features with same confirmation rate)
        confirmation_rates = np.where(
            test_counts > 0, confirmation_counts / test_counts, 0.0
        )
        avg_importances = np.where(test_counts > 0, importance_sums / test_counts, 0.0)

        # Combine: confirmation rate (weighted heavily) + normalized importance (tie-breaker)
        # Scale confirmation rate to 0-1000, importance to 0-1
        max_importance = np.max(avg_importances) if np.max(avg_importances) > 0 else 1.0
        scores = (
            confirmation_rates * 1000.0 + (avg_importances / max_importance)
        ).astype(self.info.np_dtype)

        return scores

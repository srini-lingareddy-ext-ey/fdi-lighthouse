import warnings

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF, ArrayI

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)  # ignore convergence warnings


class LassoRankingNumpy(AbstractRankingMethod):
    """
    A class that implements driver ranking using Lasso Regression (pure numpy implementation).

    This class ranks drivers based on their importance determined by the Lasso Regression
    coefficients. The ranking is computed by performing multiple iterations of Lasso
    Regression, each time splitting the data into training and testing sets, and then
    counting how frequently each driver is selected across iterations.

    This is a numpy-only implementation (no sklearn dependency) for comparison with the
    sklearn-based LassoRanking method.

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
    - Uses coordinate descent optimization for L1 regularization.
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
    def name():
        return 'Lasso (Numpy)'

    @staticmethod
    def _fit_lasso(
        X: np.ndarray, y: np.ndarray, alpha: float, max_iter: int = 1000
    ) -> np.ndarray:
        """
        Fit Lasso regression using coordinate descent optimization.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples x n_features)
        y : np.ndarray
            Target values (n_samples,)
        alpha : float
            Regularization parameter
        max_iter : int
            Maximum iterations for coordinate descent

        Returns
        -------
        np.ndarray
            Lasso coefficients (n_features,)
        """
        n_samples, n_features = X.shape

        # Center X and y
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Normalize features by sqrt(sum of squares) - matching sklearn
        X_scale = np.sqrt(np.sum(X_centered**2, axis=0) / n_samples)
        X_scale[X_scale == 0] = 1.0  # Avoid division by zero
        X_normalized = X_centered / X_scale

        # Initialize coefficients
        coef = np.zeros(n_features)

        # Coordinate descent with proper soft thresholding
        tol = 1e-4

        for _ in range(max_iter):
            coef_old = coef.copy()

            for j in range(n_features):
                # Compute residual without feature j
                residual = (
                    y_centered - X_normalized @ coef + X_normalized[:, j] * coef[j]
                )

                # Compute correlation (matching sklearn's implementation)
                rho = (X_normalized[:, j] @ residual) / n_samples

                # Soft thresholding
                if rho > alpha:
                    coef[j] = rho - alpha
                elif rho < -alpha:
                    coef[j] = rho + alpha
                else:
                    coef[j] = 0.0

            # Check convergence
            if np.max(np.abs(coef - coef_old)) < tol:
                break

        # Rescale coefficients back to original scale
        coef = coef / X_scale

        return coef

    @staticmethod
    def _train_test_split(
        X: np.ndarray, y: np.ndarray, test_size: float, random_state: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split arrays into random train and test subsets.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        test_size : float
            Proportion of dataset to include in test split (0.0 to 1.0)
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        rng = np.random.RandomState(random_state)
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)

        # Generate random permutation
        indices = rng.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

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
        - In each iteration, the data is split into training and testing sets.
        - A Lasso Regression model is fitted to the training data using coordinate
          descent optimization, and the coefficients are used to determine selected features.
        - The frequency of selection of each driver is calculated by summing the
          selection counts across all iterations.
        """
        # Normalize driver data for more stable Lasso regression results
        drivers_norm: ArrayF = stats.normalize_arr(self.info.drivers.arr).astype(
            self.info.np_dtype
        )

        # Initialize array to track which drivers are selected in each iteration
        arr_lasso: ArrayI = np.zeros(
            (self.method_params.n_iter, self.info.drivers.arr.shape[0]), dtype=np.int32
        )

        # Run multiple iterations of Lasso regression with different random states
        for k in range(self.method_params.n_iter):
            # Split data into training and testing sets (80/20 split)
            # Transpose drivers_norm to get features as columns (expected format)
            X_train, X_test, y_train, y_test = self._train_test_split(  # pyright: ignore[reportUnusedVariable]
                drivers_norm.T,
                self.info.account.arr,
                test_size=0.2,
                random_state=k,  # Different split each iteration
            )

            # Fit Lasso regression model with regularization parameter alpha=0.1
            coef = self._fit_lasso(X_train, y_train, alpha=0.1)

            # Consider a feature selected if its coefficient is above threshold (0.1)
            selected_features = coef > 0.1

            # Record which drivers were selected in this iteration (1=selected, 0=not selected)
            arr_lasso[k] = selected_features.astype(int)

        # Sum across iterations to get selection frequency for each driver
        # Higher values indicate more important drivers
        return arr_lasso.sum(axis=0).astype(self.info.np_dtype)

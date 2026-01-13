import warnings

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod

warnings.filterwarnings('ignore', category=UserWarning)


class RidgeRankingNumpy(AbstractRankingMethod):
    """
    A numpy-only implementation of driver ranking using Ridge Regression.

    This class ranks drivers based on their importance determined by the Ridge Regression
    coefficients. The ranking is computed by performing multiple iterations of Ridge
    Regression, each time splitting the data into training and testing sets, and then
    averaging the absolute coefficient values across iterations.

    Uses the closed-form solution for Ridge regression: β = (X^T X + αI)^(-1) X^T y

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

    Notes
    -----
    - Uses numpy's linear algebra for the closed-form Ridge solution
    - The method performs multiple iterations, each with a different random split of data
    - Driver importance is determined by averaging absolute coefficient values across iterations
    - No sklearn dependency required
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
        return 'Ridge_Numpy'

    @staticmethod
    def _fit_ridge(X: ArrayF, y: ArrayF, alpha: float) -> ArrayF:
        """
        Fit Ridge regression using closed-form solution.

        Ridge regression minimizes: ||y - Xβ||² + α||β||²
        Closed-form solution: β = (X^T X + αI)^(-1) X^T y

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Target vector (n_samples,).
        alpha : float
            Ridge regularization parameter.

        Returns
        -------
        np.ndarray
            Coefficient vector (n_features,).
        """
        _, n_features = X.shape

        # Center X and y (subtract means)
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Compute X^T X
        XtX = X_centered.T @ X_centered

        # Add ridge penalty: X^T X + αI
        XtX_ridge = XtX + alpha * np.eye(n_features)

        # Compute X^T y
        Xty = X_centered.T @ y_centered

        # Solve: (X^T X + αI) β = X^T y
        # Using numpy's solve is more stable than computing the inverse
        coef = np.linalg.solve(XtX_ridge, Xty)

        return coef

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
        - In each iteration, the data is split into training and testing sets (80/20 split).
        - A Ridge Regression model is fitted to the training data using the closed-form solution.
        - The absolute coefficient values are used to determine driver importance.
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

        # Set random seed for reproducibility
        rng = np.random.RandomState(42)

        # Run multiple iterations of Ridge regression with different random states
        for k in range(self.method_params.n_iter):
            # Split data into training and testing sets (80/20 split)
            # Transpose drivers_norm to get features as columns
            X = drivers_norm.T
            y = self.info.account.arr

            # Create train/test split using numpy
            n_samples = X.shape[0]
            n_train = int(0.8 * n_samples)

            # Set seed for this iteration
            rng.seed(k)
            indices = rng.permutation(n_samples)
            train_idx = indices[:n_train]

            X_train = X[train_idx]
            y_train = y[train_idx]

            # Fit Ridge regression using closed-form solution
            coef = self._fit_ridge(X_train, y_train, alpha=self.method_params.alpha)

            # Store absolute coefficient values for this iteration
            # Absolute values ensure we capture both positive and negative importance
            arr_ridge[k] = np.abs(coef)

        # Average absolute coefficient values across iterations to get final ranking scores
        # Higher values indicate more important drivers
        return arr_ridge.mean(axis=0).astype(self.info.np_dtype)

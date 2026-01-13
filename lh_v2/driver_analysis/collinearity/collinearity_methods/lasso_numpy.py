"""
LASSO-based collinearity detection method (NumPy-only implementation).

This module implements a collinearity detection method using LASSO (Least Absolute
Shrinkage and Selection Operator) regression without sklearn dependencies. For each
driver, LASSO regression is performed using all other drivers as predictors. The
absolute values of the LASSO coefficients indicate the strength of the relationship
between drivers - higher coefficients suggest stronger collinearity.

Pure NumPy implementation using coordinate descent optimization.
"""

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class LassoCollinearityNumpy(AbstractCollinearityMethod):
    """
    LASSO-based collinearity detection method (NumPy-only).

    This method detects collinearity by fitting LASSO regression models where each
    driver is predicted by all other drivers. Uses coordinate descent optimization
    with L1 regularization. Pure NumPy implementation - no sklearn dependency.

    The algorithm:
    1. Normalize all drivers (z-score: mean=0, std=1 per driver)
    2. For each driver i:
       - Use all other drivers as predictors
       - Fit LASSO regression with regularization parameter alpha
       - Extract coefficient magnitudes using coordinate descent
    3. Build coefficient matrix from all regressions
    4. Symmetrize by averaging with transpose: (C + C^T) / 2
    5. Normalize to [0,1] range

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        Information about the drivers to analyze for collinearity.
    method_params : cl_params.LassoCollinearityParams
        Configuration parameters including alpha (regularization strength).

    Attributes
    ----------
    info : dts.DriverGroup
        Stored driver information.
    method_params : cl_params.LassoCollinearityParams
        LASSO-specific parameters.
    """

    def __init__(
        self,
        drivers_info: dts.DriverGroup,
        method_params: cl_params.LassoCollinearityParams,
    ):
        """
        Initialize the LASSO collinearity method.

        Parameters
        ----------
        drivers_info : dts.DriverGroup
            Driver data with shape (n_drivers, n_timeperiods).
        method_params : cl_params.LassoCollinearityParams
            Parameters including alpha for LASSO regularization.
        """
        super().__init__(drivers_info, method_params)
        self.method_params: cl_params.LassoCollinearityParams = method_params

    @staticmethod
    def name() -> str:
        """
        Get the name of this collinearity method.

        Returns
        -------
        str
            The method name 'lasso_numpy'.
        """
        return 'lasso_numpy'

    @staticmethod
    def _fit_lasso(X: ArrayF, y: ArrayF, alpha: float, max_iter: int = 5000) -> ArrayF:
        """
        Fit LASSO regression using coordinate descent (pure numpy).

        Implements L1-regularized linear regression using coordinate descent with
        soft thresholding. Matches sklearn's Lasso implementation behavior.

        Parameters
        ----------
        X : ArrayF
            Feature matrix (n_samples x n_features)
        y : ArrayF
            Target values (n_samples,)
        alpha : float
            Regularization parameter (L1 penalty strength)
        max_iter : int
            Maximum iterations for coordinate descent

        Returns
        -------
        ArrayF
            LASSO coefficients (n_features,)

        Notes
        -----
        - Uses coordinate descent optimization
        - Features are centered and scaled before fitting
        - Soft thresholding for L1 penalty: coef = sign(rho) * max(|rho| - alpha, 0)
        - Converges when max change in coefficients < 1e-4
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
        coef = np.zeros(n_features, dtype=np.float32)

        # Coordinate descent with soft thresholding
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

    def apply(self) -> ArrayF:
        """
        Apply LASSO-based collinearity detection using pure numpy.

        For each driver, fits a LASSO regression using all other drivers as
        predictors. The coefficient magnitudes indicate collinearity strength.
        Uses coordinate descent optimization for L1 regularization.

        Returns
        -------
        ArrayF
            Symmetric collinearity matrix of shape (n_drivers, n_drivers).
            Values are in range [0, 1] where:
            - 0 = no collinearity detected
            - 1 = maximum collinearity detected
            - Diagonal elements are 0 (driver with itself)

        Notes
        -----
        - Each driver is normalized independently before regression
        - The matrix is symmetrized to ensure mutual collinearity scores
        - Alpha parameter controls sparsity: higher alpha = fewer non-zero coefficients
        - Pure NumPy implementation using coordinate descent
        """
        # Get driver data: shape (n_drivers, n_timeperiods)
        drivers = self.info.arr
        n_drivers = drivers.shape[0]

        # Normalize each driver independently (z-score normalization per row)
        drivers_normalized = stats.normalize_arr(drivers)

        # Initialize coefficient matrix
        coef_matrix = np.zeros((n_drivers, n_drivers), dtype=np.float32)

        # Get alpha parameter
        alpha = self.method_params.alpha

        # For each driver, fit LASSO using all others as predictors
        for i in range(n_drivers):
            # Target driver (what we're predicting)
            y = drivers_normalized[i]

            # Predictor drivers (all others)
            # Create boolean mask for all drivers except i
            mask = np.ones(n_drivers, dtype=bool)
            mask[i] = False
            X = drivers_normalized[mask].T  # Shape: (n_timeperiods, n_drivers-1)

            # Fit LASSO regression using numpy coordinate descent
            coef = self._fit_lasso(X, y, alpha=alpha, max_iter=5000)

            # Extract coefficients and place in matrix
            # coef has shape (n_drivers-1,)
            # We need to map these back to the full driver indices
            coef_matrix[i, mask] = np.abs(coef)

        # Symmetrize the matrix: average with transpose
        # This ensures that collinearity(i,j) = collinearity(j,i)
        coef_matrix = (coef_matrix + coef_matrix.T) / 2.0

        # Set diagonal to zero (driver with itself)
        np.fill_diagonal(coef_matrix, 0.0)

        # Normalize to [0, 1] range
        max_val = coef_matrix.max()
        if max_val > 0:
            coef_matrix = coef_matrix / max_val

        return coef_matrix

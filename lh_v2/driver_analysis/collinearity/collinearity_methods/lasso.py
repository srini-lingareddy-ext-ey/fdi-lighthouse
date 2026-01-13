"""
LASSO-based collinearity detection method.

This module implements a collinearity detection method using LASSO (Least Absolute
Shrinkage and Selection Operator) regression. For each driver, LASSO regression is
performed using all other drivers as predictors. The absolute values of the LASSO
coefficients indicate the strength of the relationship between drivers - higher
coefficients suggest stronger collinearity.

The method creates a coefficient matrix where entry (i,j) represents how strongly
driver j is used to predict driver i. The matrix is then symmetrized by averaging
with its transpose to ensure mutual collinearity scores.
"""

import numpy as np
from sklearn.linear_model import Lasso

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class LassoCollinearity(AbstractCollinearityMethod):
    """
    LASSO-based collinearity detection method.

    This method detects collinearity by fitting LASSO regression models where each
    driver is predicted by all other drivers. The LASSO coefficients reveal which
    drivers are collinear - if driver j has a large coefficient when predicting
    driver i, they are considered collinear.

    The algorithm:
    1. Normalize all drivers (z-score: mean=0, std=1 per driver)
    2. For each driver i:
       - Use all other drivers as predictors
       - Fit LASSO regression with regularization parameter alpha
       - Extract coefficient magnitudes
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
            The method name 'lasso'.
        """
        return 'lasso'

    def apply(self) -> ArrayF:
        """
        Apply LASSO-based collinearity detection.

        For each driver, fits a LASSO regression using all other drivers as
        predictors. The coefficient magnitudes indicate collinearity strength.

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

            # Fit LASSO regression
            lasso = Lasso(alpha=alpha, max_iter=5000, tol=1e-4, random_state=42)
            lasso.fit(X, y)

            # Extract coefficients and place in matrix
            # lasso.coef_ has shape (n_drivers-1,)
            # We need to map these back to the full driver indices
            coef_matrix[i, mask] = np.abs(lasso.coef_)

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

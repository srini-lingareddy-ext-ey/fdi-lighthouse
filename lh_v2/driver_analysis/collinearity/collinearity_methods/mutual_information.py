import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class MutualInformationCollinearity(AbstractCollinearityMethod):
    """
    A class to compute collinearity among drivers using Mutual Information (MI).

    This class calculates the collinearity matrix based on mutual information,
    which measures the statistical dependence between two variables. Unlike
    correlation methods, MI can capture both linear and non-linear dependencies.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        An object containing information about the drivers, including their data arrays.
    method_params : cl_params.MICollinearityParams
        Parameters specific to the MI collinearity calculation method, including
        the number of bins for histogram discretization.

    Methods
    -------
    name() -> str
        Returns the name of the collinearity method.
    _mutual_information(x, y) -> float
        Computes mutual information between two 1D arrays using histogram-based approach.
    apply() -> np.ndarray
        Computes the collinearity matrix using mutual information.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the collinearity matrix based on MI.
        The shape of the array is (n, n), where `n` is the number of drivers.

    Notes
    -----
    - Mutual information is always non-negative: MI(X,Y) >= 0
    - MI(X,Y) = 0 if and only if X and Y are independent
    - MI is symmetric: MI(X,Y) = MI(Y,X)
    - The matrix is normalized to [0,1] by dividing by the maximum MI value
    - Uses histogram-based discretization with configurable number of bins
    - Pure NumPy implementation - no sklearn dependency
    """

    def __init__(
        self,
        drivers_info: dts.DriverGroup,
        method_params: cl_params.MICollinearityParams,
    ):
        assert isinstance(method_params, cl_params.MICollinearityParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{cl_params.MICollinearityParams}, Params were of type {type(method_params)}'
        )
        self.info = drivers_info
        self.method_params = method_params
        return

    @staticmethod
    def name():
        return 'MutualInformation'

    def _mutual_information(self, x: ArrayF, y: ArrayF) -> float:
        """
        Compute mutual information between two continuous variables using histograms.

        MI(X,Y) = ∑∑ p(x,y) * log(p(x,y) / (p(x) * p(y)))

        Parameters
        ----------
        x : ArrayF
            First variable (1D array).
        y : ArrayF
            Second variable (1D array).

        Returns
        -------
        float
            Mutual information value (non-negative).

        Notes
        -----
        - Uses 2D histogram to estimate joint probability distribution
        - Marginal distributions computed from joint distribution
        - Handles zero probabilities to avoid log(0) errors
        - More bins = better resolution but requires more data
        """
        # Create 2D histogram for joint distribution
        hist_2d, _, _ = np.histogram2d(
            x, y, bins=self.method_params.n_bins, density=False
        )

        # Normalize to get probabilities
        p_xy = hist_2d / np.sum(hist_2d)

        # Compute marginal distributions
        p_x = np.sum(p_xy, axis=1)  # Sum over y
        p_y = np.sum(p_xy, axis=0)  # Sum over x

        # Outer product for independence assumption: p(x) * p(y)
        p_x_p_y = p_x[:, np.newaxis] * p_y[np.newaxis, :]

        # Compute MI, avoiding log(0) by masking zero probabilities
        mask = (p_xy > 0) & (p_x_p_y > 0)
        mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / p_x_p_y[mask]))

        return float(mi)

    def apply(self) -> ArrayF:
        """
        Compute the mutual information collinearity matrix for all driver pairs.

        This method normalizes the driver data, computes pairwise mutual information
        for all driver combinations, and returns a symmetric matrix where each element
        represents the statistical dependence between two drivers.

        Returns
        -------
        ArrayF
            A 2D array of shape (n_drivers, n_drivers) containing normalized MI values
            in the range [0, 1]. Diagonal elements are set to zero.

        Notes
        -----
        - Each driver is normalized independently before computing MI
        - Matrix is symmetric: MI[i,j] = MI[j,i]
        - Diagonal is set to zero (self-collinearity excluded)
        - Values normalized by dividing by maximum MI to get [0,1] range
        - Higher values indicate stronger statistical dependence (collinearity)
        """
        # Normalize driver data (z-score: mean=0, std=1 per driver)
        drivers_norm = np.astype(
            np.apply_along_axis(stats.normalize, 1, self.info.arr), self.info.np_dtype
        )

        # Initialize MI matrix
        n_drivers = self.info.arr.shape[0]
        mi_matrix = np.zeros((n_drivers, n_drivers), self.info.np_dtype)

        # Compute MI for all pairs (upper triangle only, then symmetrize)
        for i in range(n_drivers):
            for j in range(i + 1, n_drivers):
                mi = self._mutual_information(drivers_norm[i], drivers_norm[j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi  # Symmetric

        # Normalize to [0, 1] range by dividing by max value
        max_mi = np.max(mi_matrix)
        if max_mi > 0:
            mi_matrix = mi_matrix / max_mi

        # Set diagonal to zero (exclude self-collinearity)
        np.fill_diagonal(mi_matrix, 0)

        return mi_matrix

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class VIFCollinearity(AbstractCollinearityMethod):
    """
    A class to compute collinearity among drivers using Variance Inflation Factor (VIF).

    This class calculates the collinearity matrix based on the Variance Inflation Factor (VIF),
    which quantifies the severity of multicollinearity in regression analysis.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        An object containing information about the drivers, including their data arrays.
    method_params : cl_params.VIFCollinearityParams
        Parameters specific to the VIF collinearity calculation method.

    Methods
    -------
    name() -> str
        Returns the name of the collinearity method.
    apply() -> np.ndarray
        Computes the collinearity matrix using the VIF method.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the collinearity matrix based on VIF. The shape of the
        array is (n, n), where `n` is the number of rows (or columns) in `self.info.arr`.

    Notes
    -----
    - The VIF is calculated for each pair of drivers using the R-squared value obtained
      from regression analysis.
    - The diagonal elements of the collinearity matrix are set to zero to exclude self-collinearity.
    """

    def __init__(
        self,
        drivers_info: dts.DriverGroup,
        method_params: cl_params.VIFCollinearityParams,
    ):
        assert isinstance(method_params, cl_params.VIFCollinearityParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{cl_params.VIFCollinearityParams}, Params were of type {type(method_params)}'
        )
        self.info = drivers_info
        self.method_params = method_params
        return

    @staticmethod
    def name():
        return 'VIF'

    def apply(self) -> ArrayF:
        """
        Compute the R-squared values for pairwise collinearity between rows of a normalized array.

        This method normalizes the rows of the input array and calculates the R-squared values
        for the pairwise collinearity between the rows. The R-squared values are computed using
        subspace projections.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n, n), where n is the number of rows in the input array.
            Each element [i, j] represents the R-squared value for the collinearity between
            row i and row j of the normalized array.
        """
        drivers_norm = np.astype(
            np.apply_along_axis(stats.normalize, 1, self.info.arr), self.info.np_dtype
        )
        arr_basis = np.ones((2, self.info.arr.shape[1]), self.info.np_dtype)
        r_squared = np.zeros(
            (self.info.arr.shape[0], self.info.arr.shape[0]), self.info.np_dtype
        )
        for i in range(self.info.arr.shape[0]):
            for j in range(i + 1, self.info.arr.shape[0]):
                arr_basis[1] = drivers_norm[j]
                r_squared[i, j] = stats.rsquared(
                    stats.subspace_proj(drivers_norm[i], arr_basis), drivers_norm[i]
                )
                arr_basis[1] = drivers_norm[i]
                r_squared[j, i] = stats.rsquared(
                    stats.subspace_proj(drivers_norm[j], arr_basis), drivers_norm[j]
                )
        return r_squared

import warnings

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.ranking_types as rdt
import lh_v2.params.driver_analysis_params.ranking_params as dr_params
import lh_v2.stats as stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractRankingMethod


def last_index_larger(arr: ArrayF, val: float) -> int:
    """
    Finds the last index in the array where the value is greater than the specified value.

    Parameters
    ----------
    arr : np.ndarray
        The input array to search.
    val : float
        The value to compare against.

    Returns
    -------
    int
        The last index where the array element is greater than `val`.
        Returns -1 if no such element exists.
    """
    t_inds = np.where(arr > val)[0]
    if t_inds.size > 0:
        return int(t_inds[-1])
    else:
        return -1


class PCARanking(AbstractRankingMethod):
    """
    Principal Component Analysis (PCA) based driver ranking method.

    This class implements a driver ranking method using Principal Component Analysis (PCA).
    It ranks drivers based on their correlation with the account after projecting both
    into a reduced dimensional space determined by the principal components.

    Parameters
    ----------
    account_driver_info : dts.account_driver_types.AccountDriverGroup
        Information about the account and its associated drivers, including
        arrays of driver data and account data.
    method_params : dr_params.PCARankingParams
        Parameters for the PCA ranking method, including the dimension
        reduction method to be used.

    Attributes
    ----------
    components : np.ndarray
        Principal components computed during the ranking process.
    singular_vals : np.ndarray
        Singular values associated with the principal components.
    info : dts.account_driver_types.AccountDriverGroup
        Reordered account and driver information.
    method_params : dr_params.PCARankingParams
        Parameters for the PCA ranking method.

    Methods
    -------
    apply()
        Apply the PCA ranking method and return the ranking scores.
    name()
        Return the name of the ranking method.

    Notes
    -----
    The ranking is based on absolute Pearson correlation coefficients between
    driver projections and account projections in a reduced dimensional space.
    Two dimension reduction methods are supported:
    - ACCOUNT: Selects dimensions based on correlation with the account
    - DRIVERS: Selects dimensions based on the elbow in singular values
    """

    def __init__(
        self,
        account_driver_info: dts.AccountDriverGroup,
        method_params: dr_params.PCARankingParams,
    ):
        assert isinstance(method_params, dr_params.PCARankingParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{dr_params.PCARankingParams}, Params were of type {type(method_params)}'
        )
        sorted_drivers = sorted(account_driver_info.drivers.map.keys())
        self.info = dts.account_driver_types.AccountDriverGroup(
            account=account_driver_info.account,
            drivers=account_driver_info.drivers.order(sorted_drivers),
            np_dtype=account_driver_info.np_dtype,
        )
        self.method_params = method_params
        return

    @staticmethod
    def name() -> str:
        return 'PCA'

    def apply(self) -> ArrayF:
        """
        Applies the PCA-based driver ranking method and computes the absolute
        Pearson correlation coefficients between driver projections and the
        account projection.

        The method performs the following steps:
        1. Normalizes the driver and account data.
        2. Computes the principal components and singular values using PCA.
        3. Determines the dimensions to use based on the specified dimension
           reduction method (`PCADimMethods.ACCOUNT` or `PCADimMethods.DRIVERS`).
        4. Projects the drivers and account onto the selected subspace.
        5. Computes the Pearson correlation coefficients between the projections.

        Returns
        -------
        np.ndarray
            An array of absolute Pearson correlation coefficients between the
            driver projections and the account projection.

        Raises
        ------
        Warning
            If the singular values are too small or decrease too steadily to
            determine the correct number of dimensions when using the
            `PCADimMethods.DRIVERS` method.
        """
        # Step 1: Normalize the driver data along each row (each driver separately)
        drivers_norm = np.apply_along_axis(
            stats.normalize, 1, self.info.drivers.arr
        ).astype(self.info.np_dtype)

        # Normalize the account data
        account_norm = stats.len_norm(self.info.account.arr).astype(self.info.np_dtype)

        # Step 2: Compute principal components and singular values
        components, singular_vals = stats.pca(drivers_norm)
        # Store the results as class attributes for potential later use
        self.components, self.singular_vals = components, singular_vals

        # Step 3: Determine which dimensions to use based on the dimension reduction method
        if self.method_params.dim_method == rdt.PCADimMethods.ACCOUNT:
            # ACCOUNT method: Use dimensions most correlated with account data

            # Calculate dot products between each normalized driver and the account
            dots = np.zeros((drivers_norm.shape[0],), self.info.np_dtype)
            for k in range(drivers_norm.shape[0]):
                dots[k] = np.dot(drivers_norm[k], account_norm)

            # Select dimensions with highest absolute correlation (dot product)
            # The number of dimensions is the square root of the total number of drivers
            used_dims = np.argsort(np.abs(dots))[::-1][
                : np.int32(np.ceil(np.sqrt(dots.shape[0])))
            ].astype(np.int32)

        elif self.method_params.dim_method == rdt.PCADimMethods.DRIVERS:
            # DRIVERS method: Use the "elbow method" on singular values

            # Calculate differences between consecutive singular values
            sv_dif = np.zeros((singular_vals.shape[0] - 1,), self.info.np_dtype)
            for k in range(sv_dif.shape[0]):
                sv_dif[k] = singular_vals[k] - singular_vals[k + 1]

            # Find the last index where the difference is larger than 2 (the "elbow")
            last_ind = last_index_larger(sv_dif, 2)

            if last_ind == -1:
                # No clear elbow found, use a default number of dimensions
                warnings.warn(
                    'PCARanking: singular values were either '
                    'too small or too steadily decreasing to find correct '
                    'dimension number'
                )
                used_dims = np.array(
                    list(range(np.int32(np.ceil(np.sqrt(singular_vals.shape[0]))))),
                    dtype=np.int32,
                )
            else:
                # Use dimensions up to the elbow
                used_dims = np.array(list(range(last_ind)), dtype=np.int32)
        else:
            # Handle unexpected dimension reduction method
            raise ValueError(
                'Given dim_method is not a possible dim_method. '
                'This error should never occur since it should '
                'be found when something was cast to a PCADimMethods type.'
            )

        # Step 4: Project drivers and account data onto the selected subspace
        # Project each normalized driver onto the selected principal components
        driver_projs = np.apply_along_axis(
            stats.subspace_proj, 1, drivers_norm, components[used_dims]
        ).astype(self.info.np_dtype)

        # Project normalized account data onto the same subspace
        account_proj = stats.subspace_proj(account_norm, components[used_dims]).astype(
            self.info.np_dtype
        )

        # Step 5: Calculate Pearson correlation between each driver projection and account projection
        corrs = np.apply_along_axis(
            stats.pearson_correlation, 1, driver_projs, account_proj
        ).astype(self.info.np_dtype)

        # Return absolute values of correlations (both positive and negative correlations are relevant)
        return np.abs(corrs)

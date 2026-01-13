import numpy as np
import sklearn.cluster as skc

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class KMeansCollinearity(AbstractCollinearityMethod):
    """
    A class to compute collinearity among drivers using K-Means clustering.

    This class calculates a collinearity matrix based on the co-occurrence of drivers
    in the same cluster across multiple K-Means clustering runs with varying cluster sizes
    and random seeds.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        An object containing information about the drivers, including their data arrays.
    method_params : cl_params.KMeansCollinearityParams
        Parameters specific to the K-Means collinearity calculation method, such as the number of seeds.

    Methods
    -------
    name() -> str
        Returns the name of the collinearity method
    apply() -> np.ndarray
        Computes the collinearity matrix using K-Means clustering.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the collinearity matrix based on K-Means clustering.
        The shape of the array is (n, n), where `n` is the number of rows (or columns) in
        `self.info.arr`.

    Notes
    -----
    - The input data is normalized before clustering.
    - The co-occurrence matrix is computed by counting how often pairs of drivers are
      assigned to the same cluster across multiple clustering runs.
    - The diagonal elements of the collinearity matrix are set to zero to exclude
      self-collinearity.
    - The number of clusters varies from 2 to approximately two-thirds of the number of
      drivers.
    - The clustering process is repeated with multiple random seeds to ensure robustness.
    """

    def __init__(
        self,
        drivers_info: dts.DriverGroup,
        method_params: cl_params.KMeansCollinearityParams,
    ):
        assert isinstance(method_params, cl_params.KMeansCollinearityParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{cl_params.KMeansCollinearityParams}, Params were of type {type(method_params)}'
        )
        self.info = drivers_info
        self.method_params = method_params
        return

    @staticmethod
    def name():
        return 'KMeans'

    def apply(self) -> ArrayF:
        """
        Apply the collinearity analysis using KMeans clustering and compute the
        co-occurrence matrix for the drivers.

        Returns
        -------
        np.ndarray
            A 2D array representing the co-occurrence matrix, where each element
            indicates the frequency of two drivers being clustered together across
            multiple seeds and cluster sizes.
        """
        drivers_norm = np.astype(
            np.apply_along_axis(stats.normalize, 1, self.info.arr), self.info.np_dtype
        )
        cluster_sizes = range(2, int(self.info.arr.shape[0] / 1.5))
        arr_co_occurence = np.zeros(
            (self.info.arr.shape[0], self.info.arr.shape[0]), self.info.np_dtype
        )

        c = 0
        for n_clusters in cluster_sizes:
            for seed in range(self.method_params.n_seeds):
                kmeans = skc.KMeans(n_clusters=n_clusters, random_state=seed)
                labels = kmeans.fit_predict(drivers_norm)
                for driver in range(self.info.arr.shape[0]):
                    for driver2 in range(driver + 1, self.info.arr.shape[0]):
                        if driver != driver2 and labels[driver] == labels[driver2]:
                            arr_co_occurence[driver, driver2] += 1
                            arr_co_occurence[driver2, driver] += 1
                c += 1
        return arr_co_occurence / c

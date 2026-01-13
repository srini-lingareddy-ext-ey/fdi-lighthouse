from enum import Enum
from typing import Callable

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class CorrelationMethodEnum(Enum):
    """
    CorrelationMethodEnum is an enumeration that defines the types of correlation methods
    that can be used for statistical analysis.

    Attributes
    ----------
    PEARSON : str
        Represents the Pearson correlation coefficient, which measures the linear
        relationship between two datasets.
    SPEARMAN : str
        Represents the Spearman rank correlation, which assesses how well the relationship
        between two variables can be described using a monotonic function.
    KENDALL : str
        Represents the Kendall rank correlation, which measures the ordinal association
        between two variables.
    """

    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    KENDALL = 'kendall'


class CorrelationCollinearity(AbstractCollinearityMethod):
    """
    A class to compute collinearity among drivers using various correlation methods.

    This class calculates a weighted average of correlation matrices derived from
    multiple correlation methods (e.g., Pearson, Spearman, Kendall). The weights
    for each correlation method can be customized, and the resulting collinearity
    matrix is normalized.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        An object containing information about the drivers, including their data arrays.
    method_params : cl_params.CorrelationCollinearityParams, optional
        A parameter object that specifies configuration for the correlation collinearity method,
        including weights for different correlation methods (pearson_weight, spearman_weight,
        kendall_weight). Default is an instance of CorrelationCollinearityParams with default values.

    Attributes
    ----------
    weights : dict[CorrelationMethodEnum, float]
        A dictionary containing the normalized weights for each correlation method.

    Methods
    -------
    name()
        Returns the name of the collinearity method.
    _correlation_fun_map()
        Returns a mapping of correlation methods to their corresponding functions.
    apply()
        Computes the weighted average collinearity matrix using the specified correlation methods.

    Raises
    ------
    ValueError
        If the provided weights contain unsupported correlation methods.
    AssertionError
        If the sum of the weights is zero or if a specified method is not supported.
    Exception
        For any other unexpected errors during initialization or computation.
    """

    def __init__(
        self,
        drivers_info: dts.DriverGroup,
        method_params: cl_params.CorrelationCollinearityParams,
    ):
        assert isinstance(method_params, cl_params.CorrelationCollinearityParams), (
            f'{self.name()} Method Params not of correct type, Expected '
            f'{cl_params.CorrelationCollinearityParams}, Params were of type {type(method_params)}'
        )
        self.info = drivers_info
        self.method_params = method_params
        self.weights = {
            CorrelationMethodEnum.PEARSON: self.method_params.pearson_weight,
            CorrelationMethodEnum.SPEARMAN: self.method_params.spearman_weight,
            CorrelationMethodEnum.KENDALL: self.method_params.kendall_weight,
        }
        if not np.isclose(np.array([val for val in self.weights.values()]).sum(), 1):
            total = float(np.array([val for val in self.weights.values()]).sum())
            assert not np.isclose(total, 0), 'Correlation Method Weights sum to 0.'
            for key in self.weights.keys():
                self.weights[key] /= total
        return

    @staticmethod
    def name():
        return 'Correlation'

    @staticmethod
    def _correlation_fun_map() -> dict[
        CorrelationMethodEnum, Callable[[ArrayF, ArrayF], float]
    ]:
        """
        Returns a mapping of correlation methods to their corresponding functions.

        This method provides a dictionary that maps each correlation method
        defined in the `CorrelationMethodEnum` to its respective statistical
        function for computing correlations.

        Returns
        -------
        dict[CorrelationMethodEnum, Callable[[np.ndarray, np.ndarray], float]]
            A dictionary where the keys are correlation method enums and the
            values are functions that compute the correlation between two
            NumPy arrays.
        """
        return {
            CorrelationMethodEnum.PEARSON: stats.pearson_correlation,
            CorrelationMethodEnum.SPEARMAN: stats.spearman_correlation,
            CorrelationMethodEnum.KENDALL: stats.kendalltau_correlation,
        }

    def apply(self) -> ArrayF:
        """
        Compute a weighted average of correlation matrices using specified methods.

        This method calculates correlation matrices for each method specified in
        `self.weights`, applies weights to these matrices, and computes a weighted
        average of the results. The diagonal elements of the correlation matrices
        are set to zero before averaging.

        Returns
        -------
        np.ndarray
            A 2D numpy array representing the weighted average of the correlation
            matrices. The shape of the array is (n, n), where `n` is the number of
            rows (or columns) in `self.info.arr`.

        Notes
        -----
        - The correlation methods are retrieved from `_correlation_fun_map()`.
        - The weights for each method are specified in `self.weights`.
        - The diagonal elements of the correlation matrices are set to zero to
          exclude self-correlations from the averaging process.
        """
        methods = list(self.weights.keys())
        correlations: ArrayF = np.zeros(
            (len(methods), self.info.arr.shape[0], self.info.arr.shape[0]),
            self.info.np_dtype,
        )
        for i in range(correlations.shape[0]):
            for j in range(self.info.arr.shape[0]):
                for k in range(self.info.arr.shape[0]):
                    correlations[i, j, k] = np.abs(
                        CorrelationCollinearity._correlation_fun_map()[methods[i]](
                            self.info.arr[j], self.info.arr[k]
                        )
                    )
        weights = np.zeros((correlations.shape[0]), self.info.np_dtype)
        for k in range(correlations.shape[0]):
            weights[k] = self.weights[methods[k]]
        inds = np.arange(correlations.shape[1])
        correlations[
            :,
            inds,
            inds,
        ] = 0
        return np.average(correlations, axis=0, weights=weights).astype(
            self.info.np_dtype
        )

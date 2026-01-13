import time
from collections.abc import Sequence

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.datatypes.driver_analysis_types.collinearity_types as cdt
import lh_v2.params as params
from lh_v2.shared import ArrayF
from lh_v2.util import get_logger

from .collinearity_methods import (
    AbstractCollinearityMethod,
    CorrelationCollinearity,
    HierarchicalClusteringCollinearity,
    KMeansCollinearity,
    LassoCollinearity,
    MutualInformationCollinearity,
    VIFCollinearity,
)

logger = get_logger(__name__)

COLLINEARITY_METHOD_MAPPING: dict[
    cdt.CollinearityMethodEnum, type[AbstractCollinearityMethod]
] = {
    cdt.CollinearityMethodEnum.CORRELATION: CorrelationCollinearity,
    cdt.CollinearityMethodEnum.VIF: VIFCollinearity,
    cdt.CollinearityMethodEnum.KMEANS: KMeansCollinearity,
    cdt.CollinearityMethodEnum.MUTUAL_INFORMATION: MutualInformationCollinearity,
    cdt.CollinearityMethodEnum.LASSO: LassoCollinearity,
    cdt.CollinearityMethodEnum.HIERARCHICAL_CLUSTERING: HierarchicalClusteringCollinearity,
}


def get_collinearity_method_weights() -> dict[cdt.CollinearityMethodEnum, float]:
    return {
        cdt.CollinearityMethodEnum.CORRELATION: 0.25,
        cdt.CollinearityMethodEnum.VIF: 0.25,
        cdt.CollinearityMethodEnum.KMEANS: 0.15,
        cdt.CollinearityMethodEnum.MUTUAL_INFORMATION: 0.15,
        cdt.CollinearityMethodEnum.LASSO: 0.05,
        cdt.CollinearityMethodEnum.HIERARCHICAL_CLUSTERING: 0.15,
    }


def use_driver(
    arr_collinearity: ArrayF, driver_ind: int, used_driver_inds: list[int]
) -> bool:
    """
    Determines if a driver should be used based on its collinearity with already used drivers.
    Used for pure statistical collinearity analysis.

    Parameters
    ----------
    arr_collinearity : np.ndarray
        The collinearity matrix.
    driver_ind : int
        The index of the driver to be evaluated.
    used_driver_inds : list[int]
        The indices of the drivers that have already been used.

    Returns
    -------
    bool
        True if the driver should be used, False otherwise.
    """
    # If no drivers have been used yet, always include this driver
    if len(used_driver_inds) == 0:
        return True

    # Calculate the average collinearity between this driver and all already used drivers
    col_val = float(arr_collinearity[driver_ind, used_driver_inds].mean())

    # Return True if the average collinearity is below the threshold
    # The threshold decreases as more drivers are added (1/sqrt(n) relationship)
    # This ensures that as we add more drivers, we become more selective
    return col_val < (0.7 / np.sqrt(len(used_driver_inds)))


def use_driver_pruning(
    arr_collinearity: ArrayF,
    driver_ind: int,
    used_driver_inds: list[int],
) -> bool:
    """
    Determines if a driver should be allowed based on its collinearity with already allowed drivers.
    Used for collinearity pruning of driver for LLM selection.

    Parameters
    ----------
    arr_collinearity : np.ndarray
        The collinearity matrix.
    driver_ind : int
        The index of the driver to be evaluated.
    used_driver_inds : list[int]
        The indices of the drivers that have already been used.

    Returns
    -------
    bool
        True if the driver should be allowed, False otherwise.
    """
    # If no drivers have been used yet, always include this driver
    if len(used_driver_inds) == 0:
        return True

    # Calculate the average collinearity between this driver and all already used drivers
    col_val = float(arr_collinearity[driver_ind, used_driver_inds].mean())

    # Return True if the average collinearity is below the threshold
    # The threshold decreases as more drivers are added (1/log(n) relationship)
    # This ensures that as we add more drivers, we become more selective
    return col_val < (0.85 / (np.log(len(used_driver_inds)) + 1))


def select_methods(
    collinearity_params: params.CollinearityParams,
) -> list[cdt.CollinearityMethodEnum]:
    """
    Select which collinearity methods to use based on configuration parameters.

    Parameters
    ----------
    collinearity_params : params.CollinearityParams
        Configuration parameters controlling which collinearity methods to use.

    Returns
    -------
    list[cdt.CollinearityMethodEnum]
        List of collinearity methods to apply.

    Raises
    ------
    ValueError
        If both `b_remove` and `b_selected` are True, or if the configuration
        is invalid.

    Notes
    -----
    The function has three modes of operation:
    - If neither `b_remove` nor `b_selected` is True, all available methods are used.
    - If `b_remove` is True, specified methods are removed from all available methods.
    - If `b_selected` is True, only the specified methods are used.
    """
    # Check for invalid configuration: both flags cannot be True simultaneously
    if collinearity_params.b_remove and collinearity_params.b_selected:
        raise ValueError(
            'b_remove and b_selected cannot both be true. Must select one or neither to be True.'
        )

    # Default behavior: use all available collinearity methods
    elif not collinearity_params.b_remove and not collinearity_params.b_selected:
        selected_methods = [k for k in cdt.CollinearityMethodEnum]

    # Remove specified methods from all available methods
    elif collinearity_params.b_remove:
        base_list = [k for k in cdt.CollinearityMethodEnum]  # Start with all methods
        for method in collinearity_params.methods_removed:
            base_list.remove(method)  # Remove unwanted methods
        selected_methods = base_list

    # Use only specifically selected methods
    elif collinearity_params.b_selected:
        selected_methods = list(collinearity_params.methods_selected)
    else:
        raise ValueError(
            'Collinarity General Params not configured properly, '
            'this should not happen, if it does talk to a developer.'
        )

    logger.debug(f'Selected Collinearity Methods: {selected_methods}')
    assert len(selected_methods) > 0, (
        'Must use at least one Collinearity method, check config file.'
    )

    return selected_methods


def run_methods(
    drivers_info: dts.DriverGroup,
    selected_methods: Sequence[cdt.CollinearityMethodEnum],
    collinearity_params: params.CollinearityParams,
) -> ArrayF:
    """
    Executes multiple collinearity detection methods and combines their results.

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        Group containing driver data and metadata.
    selected_methods : Sequence[cdt.CollinearityMethodEnum]
        Sequence of collinearity methods to apply.
    collinearity_params : params.CollinearityParams
        Configuration parameters for collinearity methods.

    Returns
    -------
    ArrayF
        Combined collinearity matrix of shape (n_drivers, n_drivers) where each
        element represents the weighted average collinearity score between pairs
        of drivers.

    Raises
    ------
    Exception
        If any collinearity method fails during execution.

    Notes
    -----
    - Retrieves and normalizes method weights to sum to 1.0.
    - Instantiates each selected collinearity method.
    - Applies each method to generate individual collinearity matrices.
    - Combines matrices using weighted averaging.
    """
    # Assign weights to each collinearity method
    method_weights = get_collinearity_method_weights()
    method_weights = {method: method_weights[method] for method in selected_methods}

    # Check weight sum to 1, scale sum to 1 if not
    total = 0
    for method in selected_methods:
        total += method_weights[method]
    if not np.isclose(total, 1):
        for method in selected_methods:
            method_weights[method] = method_weights[method] / total

    # Create instances of each collinearity method
    method_instances: list[AbstractCollinearityMethod] = []
    for method in selected_methods:
        method_instances.append(
            COLLINEARITY_METHOD_MAPPING[method](
                drivers_info=drivers_info,
                method_params=collinearity_params[method],
            )
        )

    # Initialize 3D array to store collinearity matrices from each method
    arr_collinearity_methods: ArrayF = np.zeros(
        (
            len(selected_methods),
            drivers_info.arr.shape[0],
            drivers_info.arr.shape[0],
        ),
    )

    # Apply each collinearity method and store results
    last_time = time.time()
    for k, method in enumerate(method_instances):
        try:
            arr_collinearity_methods[k] = method.apply()
        except Exception as exc:
            print(f'Method that caused the problem - {method.name()}')
            raise exc

        logger.timing(
            f'Time of Collinearity method `{method.name()}` - {time.time() - last_time:.4f} seconds.'
        )
        last_time = time.time()

    # Initialize the final combined collinearity matrix
    arr_collinearity: ArrayF = np.zeros(
        (
            arr_collinearity_methods.shape[1],
            arr_collinearity_methods.shape[2],
        ),
    )

    # Calculate weighted average of all method matrices
    for k in range(arr_collinearity_methods.shape[0]):
        arr_collinearity += (
            arr_collinearity_methods[k] * method_weights[selected_methods[k]]
        )

    return arr_collinearity

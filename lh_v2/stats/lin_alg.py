import numpy as np

from lh_v2.shared import ArrayF

from .normalize_arrs import normalize_arr


def subspace_proj(vector: ArrayF, subspace_basis: ArrayF) -> ArrayF:
    """
    Calculates the projection of a vector onto a subspace.

    Parameters
    ----------
    vector : (np.ndarray)
        The vector to be projected into the basis.
    subspace_basis : (np.ndarray)
        A matrix whose columns form a basis for the subspace.

    Returns
    -------
    np.ndarray
        The projection of the vector onto the subspace.
    """
    # Calculate the projection matrix
    P = np.linalg.pinv(subspace_basis) @ subspace_basis
    # Project the vector
    return P @ vector


def least_squares(input: ArrayF, output: ArrayF) -> ArrayF:
    return np.linalg.inv(input.T @ input) @ input.T @ output


def pca(matrix: ArrayF) -> tuple[ArrayF, ArrayF]:
    matrix = normalize_arr(matrix)
    svd = np.linalg.svd(matrix)
    return svd.Vh[: matrix.shape[0]], svd.S

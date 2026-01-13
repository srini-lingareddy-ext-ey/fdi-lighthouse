"""
Hierarchical clustering-based collinearity detection method.

This module implements collinearity detection using hierarchical clustering.
Drivers are clustered based on their similarity, and the cophenetic distance
from the dendrogram indicates how collinear they are - drivers merged early
in the dendrogram are highly collinear, while those merged late are independent.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class HierarchicalClusteringCollinearity(AbstractCollinearityMethod):
    """
    Hierarchical clustering-based collinearity detection.

    This method uses hierarchical clustering to identify collinear drivers.
    The cophenetic distance matrix from the dendrogram provides a measure of
    similarity - drivers that cluster together early (low cophenetic distance)
    are highly collinear.

    Algorithm:
    1. Normalize all driver data together (z-score: mean=0, std=1 across all values)
    2. Transpose to (n_timeperiods, n_drivers) for distance calculation
    3. Compute pairwise distances using specified metric:
       - 'correlation': 1 - |correlation coefficient|
       - 'euclidean': standard Euclidean distance
    4. Build hierarchical clustering with specified linkage method:
       - 'ward': minimize variance within clusters
       - 'complete': maximum distance between clusters
       - 'average': average distance between clusters
       - 'single': minimum distance between clusters
    5. Extract cophenetic distances (heights at which drivers merge)
    6. Invert cophenetic matrix: collinearity = max - cophenetic
    7. Normalize to [0,1] range

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        Driver data with shape (n_drivers, n_timeperiods).
    method_params : cl_params.HierarchicalClusteringCollinearityParams
        Configuration including linkage_method and distance_metric.

    Attributes
    ----------
    info : dts.DriverGroup
        Stored driver information.
    method_params : cl_params.HierarchicalClusteringCollinearityParams
        Hierarchical clustering parameters.
    """

    def __init__(
        self,
        drivers_info: dts.DriverGroup,
        method_params: cl_params.HierarchicalClusteringCollinearityParams,
    ):
        """
        Initialize hierarchical clustering collinearity method.

        Parameters
        ----------
        drivers_info : dts.DriverGroup
            Driver data with shape (n_drivers, n_timeperiods).
        method_params : cl_params.HierarchicalClusteringCollinearityParams
            Parameters for linkage method and distance metric.
        """
        super().__init__(drivers_info, method_params)
        self.method_params: cl_params.HierarchicalClusteringCollinearityParams = (
            method_params
        )

    @staticmethod
    def name() -> str:
        """
        Get the name of this collinearity method.

        Returns
        -------
        str
            The method name 'hierarchical_clustering'.
        """
        return 'hierarchical_clustering'

    def apply(self) -> ArrayF:
        """
        Apply hierarchical clustering collinearity detection.

        Builds a dendrogram from driver time series and uses cophenetic
        distances to measure collinearity. Drivers that merge early in
        the dendrogram have low cophenetic distance and are considered
        highly collinear.

        Returns
        -------
        ArrayF
            Symmetric collinearity matrix of shape (n_drivers, n_drivers).
            Values in range [0, 1] where:
            - 0 = no collinearity (merge late in dendrogram)
            - 1 = maximum collinearity (merge early in dendrogram)
            - Diagonal elements are 0

        Notes
        -----
        - Correlation distance captures linear relationships
        - Euclidean distance captures magnitude differences
        - Ward linkage tends to create balanced clusters
        - Complete linkage is sensitive to outliers
        - Cophenetic distance measures dendrogram structure stability
        """
        # Get driver data: shape (n_drivers, n_timeperiods)
        drivers = self.info.arr

        # Normalize all data together (z-score: mean=0, std=1 across entire array)
        drivers_normalized = stats.normalize(drivers)

        # Transpose to (n_timeperiods, n_drivers) for distance calculation
        # Each driver becomes a point in n_timeperiod dimensional space
        drivers_transposed = drivers_normalized.T

        # Compute pairwise distances based on metric
        if self.method_params.distance_metric == 'correlation':
            # Correlation distance: 1 - |correlation|
            # This treats positive and negative correlation as similar
            corr_matrix = np.corrcoef(drivers_transposed.T)
            # Use absolute value to treat -1 and +1 correlation as equally collinear
            distance_matrix = 1.0 - np.abs(corr_matrix)
            # Convert to condensed distance matrix for linkage
            distances = squareform(distance_matrix, checks=False)
        else:  # euclidean
            # Standard Euclidean distance in time-series space
            distances = pdist(drivers_transposed.T, metric='euclidean')

        # Build hierarchical clustering
        linkage_matrix = linkage(distances, method=self.method_params.linkage_method)

        # Compute cophenetic distances from linkage matrix
        # For each pair of samples, the cophenetic distance is the height at which
        # they first merge in the dendrogram
        n = len(drivers_normalized)
        cophenetic_matrix = np.zeros((n, n), dtype=np.float32)

        # Extract cophenetic distances from linkage matrix
        # linkage_matrix has shape (n-1, 4) where each row is [idx1, idx2, distance, sample_count]
        # We need to track at what height each pair of drivers merges
        cluster_heights = {}  # Maps cluster_id -> (height, members)

        # Initialize each driver as its own cluster
        for i in range(n):
            cluster_heights[i] = (0.0, {i})

        # Process linkage matrix to build cophenetic distances
        for i, (idx1, idx2, height, _) in enumerate(linkage_matrix):
            new_cluster_id = n + i
            idx1, idx2 = int(idx1), int(idx2)

            # Get members of each cluster being merged
            _, members1 = cluster_heights[idx1]
            _, members2 = cluster_heights[idx2]

            # Set cophenetic distance for all pairs across the two clusters
            for m1 in members1:
                for m2 in members2:
                    cophenetic_matrix[m1, m2] = height
                    cophenetic_matrix[m2, m1] = height

            # Create new cluster with combined members
            all_members = members1 | members2
            cluster_heights[new_cluster_id] = (height, all_members)

        # Convert cophenetic distances to collinearity scores
        # Low cophenetic distance = merge early = high collinearity
        # High cophenetic distance = merge late = low collinearity
        # So we invert: collinearity = max_distance - cophenetic_distance
        max_distance = cophenetic_matrix.max()
        if max_distance > 0:
            collinearity_matrix = max_distance - cophenetic_matrix
        else:
            collinearity_matrix = np.zeros_like(cophenetic_matrix)

        # Set diagonal to zero (driver with itself)
        np.fill_diagonal(collinearity_matrix, 0.0)

        # Normalize to [0, 1] range
        max_val = collinearity_matrix.max()
        if max_val > 0:
            collinearity_matrix = collinearity_matrix / max_val

        return collinearity_matrix.astype(np.float32)

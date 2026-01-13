"""
Hierarchical clustering-based collinearity detection method (NumPy-only implementation).

This module implements collinearity detection using hierarchical clustering
without scipy dependencies. Uses pure NumPy implementation of average linkage (UPGMA).
Drivers are clustered based on their similarity, and the cophenetic distance
from the dendrogram indicates how collinear they are - drivers merged early
in the dendrogram are highly collinear, while those merged late are independent.
"""

import numpy as np

import lh_v2.datatypes as dts
import lh_v2.params.driver_analysis_params.collinearity_params as cl_params
from lh_v2 import stats
from lh_v2.shared import ArrayF

from .abstract_class import AbstractCollinearityMethod


class HierarchicalClusteringCollinearityNumpy(AbstractCollinearityMethod):
    """
    Hierarchical clustering-based collinearity detection (NumPy-only).

    This method uses hierarchical clustering with average linkage to identify
    collinear drivers. Pure NumPy implementation - no scipy dependencies.
    The cophenetic distance matrix from the dendrogram provides a measure of
    similarity - drivers that cluster together early (low cophenetic distance)
    are highly collinear.

    Algorithm:
    1. Normalize all driver data together (z-score: mean=0, std=1 across all values)
    2. Transpose to (n_timeperiods, n_drivers) for distance calculation
    3. Compute pairwise distances using specified metric:
       - 'correlation': 1 - |correlation coefficient|
       - 'euclidean': standard Euclidean distance
    4. Build hierarchical clustering with average linkage (UPGMA):
       - Distance between clusters = average of all pairwise distances
       - Works well with any distance metric
    5. Extract cophenetic distances (heights at which drivers merge)
    6. Invert cophenetic matrix: collinearity = max - cophenetic
    7. Normalize to [0, 1] range

    Parameters
    ----------
    drivers_info : dts.DriverGroup
        Driver data with shape (n_drivers, n_timeperiods).
    method_params : cl_params.HierarchicalClusteringCollinearityParams
        Configuration including linkage_method and distance_metric.
        Note: linkage_method parameter is ignored - always uses average linkage.

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
            The method name 'hierarchical_clustering_numpy'.
        """
        return 'hierarchical_clustering_numpy'

    @staticmethod
    def _pairwise_distances_euclidean(X: ArrayF) -> ArrayF:
        """
        Compute condensed pairwise Euclidean distances.

        Parameters
        ----------
        X : ArrayF
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        ArrayF
            Condensed distance matrix of length n_samples * (n_samples - 1) / 2.
            Distances are ordered: (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ...
        """
        n = X.shape[0]
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                distances.append(dist)
        return np.array(distances, dtype=np.float32)

    @staticmethod
    def _square_to_condensed(square_matrix: ArrayF) -> ArrayF:
        """
        Convert square distance matrix to condensed form.

        Parameters
        ----------
        square_matrix : ArrayF
            Square symmetric distance matrix of shape (n, n).

        Returns
        -------
        ArrayF
            Condensed distance array of length n * (n - 1) / 2.
        """
        n = square_matrix.shape[0]
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(square_matrix[i, j])
        return np.array(condensed, dtype=np.float32)

    def _hierarchical_clustering_average(self, distances: ArrayF, n: int) -> ArrayF:
        """
        Perform hierarchical clustering using average linkage (UPGMA).

        Average linkage computes the distance between clusters as the average
        of all pairwise distances between members. This works well with any
        distance metric, including pre-computed correlation distances.

        Parameters
        ----------
        distances : ArrayF
            Condensed distance matrix of length n * (n - 1) / 2.
        n : int
            Number of original samples.

        Returns
        -------
        ArrayF
            Linkage matrix of shape (n-1, 4) where each row is:
            [cluster1_id, cluster2_id, distance, num_samples_in_new_cluster]
        """
        # Convert condensed distances to square matrix for easier access
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i, j] = distances[idx]
                dist_matrix[j, i] = distances[idx]
                idx += 1

        # Initialize: each sample is its own cluster
        # Track which original cluster ID maps to which actual cluster
        cluster_map = {
            i: i for i in range(n)
        }  # Maps current position -> actual cluster ID
        cluster_sizes = {i: 1 for i in range(n)}
        linkage_matrix = []

        # Current distance matrix (will be updated as clusters merge)
        current_dist = dist_matrix.copy()

        # Perform n-1 merges
        for merge_step in range(n - 1):
            # Find the pair of clusters with minimum distance
            min_dist = np.inf
            min_i, min_j = -1, -1

            for i in range(n):
                if cluster_map[i] == -1:  # Skip inactive clusters
                    continue
                for j in range(i + 1, n):
                    if cluster_map[j] == -1:  # Skip inactive clusters
                        continue
                    if current_dist[i, j] < min_dist:
                        min_dist = current_dist[i, j]
                        min_i, min_j = i, j

            # Get actual cluster IDs
            cluster_i = cluster_map[min_i]
            cluster_j = cluster_map[min_j]

            # Create new cluster ID
            new_cluster_id = n + merge_step
            new_size = cluster_sizes[cluster_i] + cluster_sizes[cluster_j]

            # Record the merge (using actual cluster IDs)
            linkage_matrix.append(
                [float(cluster_i), float(cluster_j), float(min_dist), float(new_size)]
            )

            # Update distance matrix using average linkage formula
            # For average linkage: d(k, ij) = (n_i * d(i,k) + n_j * d(j,k)) / (n_i + n_j)
            for k in range(n):
                if cluster_map[k] == -1 or k == min_i or k == min_j:
                    continue

                n_i = cluster_sizes[cluster_i]
                n_j = cluster_sizes[cluster_j]

                d_ik = current_dist[min_i, k]
                d_jk = current_dist[min_j, k]

                # Average linkage formula (weighted by cluster sizes)
                new_dist = (n_i * d_ik + n_j * d_jk) / (n_i + n_j)
                current_dist[k, min_i] = new_dist
                current_dist[min_i, k] = new_dist

            # Mark min_j as inactive, reuse min_i for the new cluster
            cluster_map[min_j] = -1
            cluster_map[min_i] = new_cluster_id
            cluster_sizes[new_cluster_id] = new_size

        return np.array(linkage_matrix, dtype=np.float32)

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
        - Average linkage computes distance as mean of all pairwise distances
        - Cophenetic distance measures dendrogram structure stability
        - Pure NumPy implementation - no scipy dependency
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
            distances = self._square_to_condensed(distance_matrix)
        else:  # euclidean
            # Standard Euclidean distance in time-series space
            distances = self._pairwise_distances_euclidean(drivers_transposed.T)

        # Build hierarchical clustering using average linkage (NumPy implementation)
        linkage_matrix = self._hierarchical_clustering_average(
            distances, len(drivers_normalized)
        )

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

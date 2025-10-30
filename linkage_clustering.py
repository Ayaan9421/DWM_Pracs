import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        if linkage not in ['single', 'complete', 'average']:
            raise ValueError("Linkage must be 'single', 'complete', or 'average'")
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        self.X = np.array(X)
        n_samples = len(X)

        # Step 1: Start with each point as its own cluster
        clusters = [[i] for i in range(n_samples)]

        # Step 2: Compute initial distance matrix
        distances = cdist(X, X)
        np.fill_diagonal(distances, np.inf)

        while len(clusters) > self.n_clusters:
            # Step 3: Find two closest clusters
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            
            # Merge clusters i and j
            new_cluster = clusters[i] + clusters[j]
            
            # Remove old clusters
            clusters.pop(max(i, j))
            clusters.pop(min(i, j))
            clusters.append(new_cluster)
            
            # Step 4: Update distance matrix
            distances = self._update_distances(distances, clusters, X)
        
        # Step 5: Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                self.labels_[point] = idx
        
        return self

    def _update_distances(self, distances, clusters, X):
        """Recompute distances between clusters after merging."""
        n = len(clusters)
        new_distances = np.full((n, n), np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self._compute_linkage_distance(clusters[i], clusters[j], X)
                new_distances[i, j] = new_distances[j, i] = d
        return new_distances

    def _compute_linkage_distance(self, cluster1, cluster2, X):
        """Compute linkage distance between two clusters."""
        points1, points2 = X[cluster1], X[cluster2]
        dists = cdist(points1, points2)
        
        if self.linkage == 'single':
            return np.min(dists)
        elif self.linkage == 'complete':
            return np.max(dists)
        elif self.linkage == 'average':
            return np.mean(dists)
    
    def plot_clusters(self):
        """Plot 2D clusters."""
        if self.X.shape[1] != 2:
            raise ValueError("Plotting only supported for 2D data.")
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels_, cmap='rainbow')
        plt.title(f"Hierarchical Clustering ({self.linkage.capitalize()} Linkage)")
        plt.show()

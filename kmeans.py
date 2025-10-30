import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        """
        n_clusters: Number of clusters
        max_iters: Maximum number of iterations
        tol: Minimum change in centroids to declare convergence
        random_state: Seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit the KMeans model to the data
        """
        if isinstance(X, pd.DataFrame):
            X = X.values  # convert DataFrame to NumPy array

        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Step 1: Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Step 2: Assign points to nearest centroid
            labels = self._assign_clusters(X)

            # Step 3: Compute new centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            # Step 4: Check for convergence
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid.
        """
        # Ensure X is a NumPy array
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        """
        Predict the closest cluster for each sample.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._assign_clusters(X)

    def fit_predict(self, X):
        """
        Fit to the data, then return the cluster labels.
        """
        self.fit(X)
        return self.predict(X)

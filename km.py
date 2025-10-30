import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import KMeans

# Load your data
# data = pd.read_csv("2D_unsupervised.csv")
# X = data[["x1", "x2"]]
# print(X.head())

X  = np.array([
        [1, 2],
        [1, 4],
        [1, 0],
        [4, 2],
        [4, 4],
        [4, 0]
])



# Run custom KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.6, marker='X')
plt.title("K-Means Clustering (From Scratch)")
plt.show()

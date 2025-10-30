import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your data
data = pd.read_csv("2D_unsupervised.csv")
X = data[["x1", "x2"]]
print(X.head())

# Run KMeans (scikit-learn)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)

# Plot results
plt.scatter(X["x1"], X["x2"], c=labels, s=40, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', s=200, alpha=0.6, marker='X')
plt.title("K-Means Clustering (Scikit-learn)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

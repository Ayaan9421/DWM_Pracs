# Example dataset
import numpy as np

from linkage_clustering import HierarchicalClustering
import pandas as pd

data = pd.read_csv("2D_unsupervised.csv")
X = data.values

# Try single linkage
hc_single = HierarchicalClustering(n_clusters=2, linkage='single')
hc_single.fit(X)
hc_single.plot_clusters()

# Try complete linkage
hc_complete = HierarchicalClustering(n_clusters=2, linkage='complete')
hc_complete.fit(X)
hc_complete.plot_clusters()

# Try average linkage
hc_average = HierarchicalClustering(n_clusters=2, linkage='average')
hc_average.fit(X)
hc_average.plot_clusters()

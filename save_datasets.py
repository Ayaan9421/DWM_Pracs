# ==========================
#  ML Visualization Datasets
#  Author: Ayaan Shaikh
# ==========================

from sklearn.datasets import make_moons, make_blobs, make_regression, make_swiss_roll
import pandas as pd
import numpy as np

# ---------- 1️⃣ 2D Supervised ----------
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
df_2d_supervised = pd.DataFrame(X, columns=['x1', 'x2'])
df_2d_supervised['label'] = y
df_2d_supervised.to_csv('2D_supervised.csv', index=False)
print("✅ Saved 2D_supervised.csv")

# ---------- 2️⃣ 2D Unsupervised ----------
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
df_2d_unsupervised = pd.DataFrame(X, columns=['x1', 'x2'])
df_2d_unsupervised.to_csv('2D_unsupervised.csv', index=False)
print("✅ Saved 2D_unsupervised.csv")

# ---------- 3️⃣ 3D Supervised ----------
X, y = make_regression(n_samples=300, n_features=3, noise=15, random_state=42)
df_3d_supervised = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
df_3d_supervised['target'] = y
df_3d_supervised.to_csv('3D_supervised.csv', index=False)
print("✅ Saved 3D_supervised.csv")

# ---------- 4️⃣ 3D Unsupervised ----------
X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
df_3d_unsupervised = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
df_3d_unsupervised['color'] = color  # optional, for coloring visualizations
df_3d_unsupervised.to_csv('3D_unsupervised.csv', index=False)
print("✅ Saved 3D_unsupervised.csv")

print("\n🎉 All 4 datasets successfully saved as CSV files!")

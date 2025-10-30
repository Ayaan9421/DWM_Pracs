from sklearn.datasets import make_moons
import pandas as pd

# Load dataset
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
data = make_moons()

# Combine features and target into a single DataFrame
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Save to CSV
df.to_csv('2D_supervised.csv', index=False)

print("✅ Iris dataset saved as '2D_supervised.csv'")
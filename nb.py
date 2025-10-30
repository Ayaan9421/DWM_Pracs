# Import the libs
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
from naive_bayes import NaiveBayes

# Load the data
data = pd.read_csv("2D_supervised.csv")
X = data.drop("label", axis=1).values
y = data["label"].values

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state = 42)

# Model Training
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Prediction and Accuracy
y_pred = nb.predict(X_test)
print("Accuracy: ", nb.accuracy(y_test, y_pred))

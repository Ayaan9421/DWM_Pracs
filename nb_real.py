# Import the libs
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

# Load the data
X, y = load_iris(return_X_y = True)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)


cr = classification_report(y_test, y_pred)
print(cr)

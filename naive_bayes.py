import numpy as np

class NaiveBayes:
	def __init__(self):
		self.classes = None
		self.mean = {}
		self.var = {}
		self.priors = {}
		
	def fit(self, X, y):
		"""
		Train the NB Classifier
		X: numpy array of shape  (n_samples, n_features)
		y: numpy array of shape (n_samples, )
		"""
		
		self.classes = np.unique(y)
		n_samples, n_features = X.shape
		
		# Compute Mean, Variance, and Priors for each class
		for c in self.classes:
			X_c = X[y==c]
			self.mean[c] = np.mean(X_c, axis = 0)
			self.var[c] = np.var(X_c, axis = 0)
			self.priors[c] = X_c.shape[0] / n_samples

	def _gaussian_pdf(self, class_idx, x):
		""" Compute the Gaussian Probability Density Function Value """
		mean = self.mean[class_idx]
		var = self.var[class_idx]
		numerator = np.exp(- (x-mean) ** 2 / (2 * var + 1e-9))
		denominator = np.sqrt(2 * np.pi * var + 1e-9)
		return numerator / denominator

	def _posterior(self, x):
		posteriors = []
		for c in self.classes:
			prior = np.log(self.priors[c])
			conditional = np.sum(np.log(self._gaussian_pdf(c , x)))
			posterior = proir = + conditional
			posteriors.append(posterior)
		return self.classes[np.argmax(posteriors)]


	def predict(self, X):
		return np.array([self._posterior(x) for x in X])
		

	def accuracy(self, y_true, y_pred):
		return np.mean(y_true == y_pred)

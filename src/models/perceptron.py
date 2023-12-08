from model import Model
import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron


class Perceptron(Model):

	def __init__(self, constants):
		super().__init__(constants)
		self.name = "Perceptron"

		# Constants
		tol = 0										# Tolerance for the stopping criterion
		eta0 = 1									# Constant by which the updates are multiplied
		penalty = None								# Regularization term, None for no regularization
		shuffle = False								# Whether or not the training data should be shuffled after each epoch
		verbose = 0									# Verbosity level

		# Create the model
		self.model = SklearnPerceptron(tol=tol, eta0=eta0, penalty=penalty, shuffle=shuffle, verbose=verbose)
	


# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.linear_model import Perceptron

# class CustomPerceptron:
# 	def __init__(self, **kwargs):
# 		self.model = Perceptron(**kwargs)
# 		self.history = {"accuracy": [], "loss": []}

# 	def train(self, X, Y, X_val, Y_val, epochs):
# 		for _ in range(epochs):
# 			self.model.partial_fit(X, Y, classes=np.unique(Y))
# 			predictions = self.model.predict(X_val)
# 			accuracy = accuracy_score(Y_val, predictions)
# 			loss = mean_squared_error(Y_val, predictions)
# 			self.history["accuracy"].append(accuracy)
# 			self.history["loss"].append(loss)
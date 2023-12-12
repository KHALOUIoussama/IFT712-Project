from model import Model
import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron


class Perceptron(Model):

	def __init__(self):
		self.name = "Perceptron"

		# Constants
		tol = 0										# Tolerance for the stopping criterion
		eta0 = 1									# Constant by which the updates are multiplied
		penalty = None								# Regularization term, None for no regularization
		shuffle = False								# Whether or not the training data should be shuffled after each epoch

		# Create the model
		self.model = SklearnPerceptron(tol=tol, eta0=eta0, penalty=penalty, shuffle=shuffle)
	
	def get_hyperparameters_choices(self):
		"""
		Return the hyperparameters to test for the model.
		"""
		return {
				"alpha": [0.1, 0.2, 0.01],
				"max_iter": [100, 200, 300],
			}
	
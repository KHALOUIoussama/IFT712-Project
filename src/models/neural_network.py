from model import Model
import numpy as np
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(Model):

	def __init__(self, constants):
		super().__init__(constants)
		self.name = 'Neural Network'

		# Parameters of the neural network
		beta_1 = 0.9				# Exponential decay rate for estimates of first moment vector in adam
		beta_2 = 0.999				# Exponential decay rate for estimates of second moment vector in adam
		epsilon = 1e-08				# Value for numerical stability in adam

		# Initialize the neural network
		self.model = MLPClassifier(activation='relu', solver='adam', shuffle=True, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon) 


	# def train(self, X, Y, hyperparameters):
	# 	"""
	# 	Train the model on the training set (X, Y).

	# 	Parameters:
	# 		- X : np.array, shape (n_samples, n_features)
	# 		- Y : np.array, shape (n_samples, )
	# 		- hyperparameters : dict, contains the hyperparameters to use and their values, output of find_optimal_hyperparameters
	# 	"""
	# 	self.model.set_params(**hyperparameters)
	# 	self.history = self.model.fit(X, Y)
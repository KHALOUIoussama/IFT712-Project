from model import Model
import numpy as np
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(Model):

	def __init__(self):
		self.name = 'Neural Network'

		# Parameters of the neural network
		beta_1 = 0.9				# Exponential decay rate for estimates of first moment vector in adam
		beta_2 = 0.999				# Exponential decay rate for estimates of second moment vector in adam
		epsilon = 1e-08				# Value for numerical stability in adam

		# Initialize the neural network
		self.model = MLPClassifier(activation='relu', solver='adam', shuffle=True, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon) 

	def get_hyperparameters_choices(self):
		"""
		Return the hyperparameters to test for the model.
		"""
		return {
				"alpha": [1e-4, 1e-5],
				"max_iter": [300, 500],
				"batch_size": ["auto"],
				"hidden_layer_sizes" : [(100,)]
			}
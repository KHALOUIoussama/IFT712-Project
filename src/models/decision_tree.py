from model import Model
import numpy as np


class DecisionTree(Model):

	def __init__(self, constants):
		super().__init__(constants)
		self.name = "DecisionTree"

	def find_optimal_hyperparameters(self, X, Y, hyperparameters, cv=5):
		pass

	def train(self, X, Y, hyperparameters):
		pass

	def predict(self, X):
		pass
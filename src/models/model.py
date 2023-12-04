from abc import ABC, abstractmethod

class Model(ABC):
	""" Abstract class for models """

	def __init__(self, name, constants):
		"""
		Parameters:
			- name : str, name of the model
			- constants : Constants, contains the number of labels, features and samples, and the labels
		"""
		self.name = name
		self.constants = constants

	@abstractmethod
	def find_optimal_hyperparameters(self, X, Y, hyperparameters, cv=5):
		""" Find the optimal hyperparameters for the model.
		Parameters:
			- X : np.array, shape (n_samples, n_features)
			- Y : np.array, shape (n_samples, n_labels), one-hot encoding
			- hyperparameters : dict, contains the hyperparameters to test. For each hyperparameter, the values to test are given in a list.   TODO : Check GridSearchCV de sklearn !!!!!!!!
			- cv : int, number of folds for cross-validation
		Output:
			- best_hyperparameters : dict, contains the optimal hyperparameters
		"""
		pass

	@abstractmethod
	def train(self, X, Y, hyperparameters):
		""" Train the model on the training set (X, Y).
		Parameters:
			- X : np.array, shape (n_samples, n_features)
			- Y : np.array, shape (n_samples, n_labels), one-hot encoding
			- hyperparameters : dict, contains the hyperparameters to use and their values, output of find_optimal_hyperparameters
		"""
		pass

	@abstractmethod
	def predict(self, X):
		""" Predict the labels of the samples in X.
		Parameters:
			- X : np.array, shape (n_samples, n_features)
		Output:
			- Y : np.array, shape (n_samples, n_labels), one-hot encoding
		"""
		pass




	


from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from random import choice


class Model(ABC):
	""" Abstract class for models """

	def __init__(self):
		self.name = None
		self.model = None
	
	@abstractmethod
	def get_hyperparameters_choices(self):
		""" Return the hyperparameters to test on the format : "hyperparameter": [values] """
		pass

	def find_optimal_hyperparameters(self, X, Y, hyperparameters_choices, cv=5):
		"""
		Find the optimal hyperparameters for the model using GridSearchCV.

		Parameters:
		- X : np.array, shape (n_samples, n_features), training features.
		- Y : np.array, shape (n_samples,), training labels.
		- hyperparameters_choices : dict, a dictionary with hyperparameters to test. Each hyperparameter key maps to a list of values to test.
		- cv : int, number of folds for cross-validation.

		Returns:
		- best_hyperparameters : dict, the best hyperparameters found.
		"""
    
		# Initialize GridSearchCV
		grid_search = GridSearchCV(estimator=self.model, param_grid=hyperparameters_choices, cv=cv, scoring='accuracy')
		
		# Execute GridSearchCV on the data
		try:
			grid_search.fit(X, Y)
		except ValueError as e:
			print("The hyperparameters are not valid ! Here is the error message :")
			print(e)
			return None
		
		# Retrieve and return the best hyperparameters
		best_hyperparameters = grid_search.best_params_
		
		return best_hyperparameters

	def get_alea_hyperparameters(self, hyperparameters_choices):
		""" Return random combination of hyperparameters (quicker than testing each combination)
		Paremeters:
			- hyperparameters_choices : dict, a dictionary with hyperparameters to test. Each hyperparameter key maps to a list of values to test.
		Returns:
			- hyperparameters : dict, the random hyperparameters
		"""
		hyperparameters = {}
		for key in hyperparameters_choices.keys():
			hyperparameters[key] = choice(hyperparameters_choices[key])
		return hyperparameters

	def train(self, X, Y, hyperparameters):
		""" Train the model on the training set (X, Y).
		Parameters:
			- X : np.array, shape (n_samples, n_features)
			- Y : np.array, shape (n_samples, )
			- hyperparameters : dict, contains the hyperparameters to use and their values, output of find_optimal_hyperparameters
		"""
		self.model.set_params(**hyperparameters)
		self.model.fit(X, Y)		

	def predict(self, X):
		""" Predict the labels of the samples in X.
		Parameters:
			- X : np.array, shape (n_samples, n_features)
		Output:
			- Y : np.array, shape (n_samples,)
		"""
		if self.model is None:
			raise ValueError("The model has not been trained yet. Please call the train method.")
		Y = self.model.predict(X)
		return Y

	def predict_proba(self, X):
		""" Predict the probabilities of each label for the samples in X.
		Parameters:
			- X : np.array, shape (n_samples, n_features)
		Output:
			- Y : np.array, shape (n_samples, n_labels)
		"""
		if self.model is None:
			raise ValueError("The model has not been trained yet. Please call the train method.")
		# Test if model has the predict_proba method
		if not hasattr(self.model, "predict_proba"):
			print("WARNING : The model has no predict_proba method !")
			return None
		Y = self.model.predict_proba(X)
		return Y
    
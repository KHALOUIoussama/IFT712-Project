from abc import ABC
from sklearn.model_selection import GridSearchCV
from random import choice


class Model(ABC):
	""" Abstract class for models """

	def __init__(self, constants):
		"""
		Parameters:
			- constants : Constants, contains the number of labels, features and samples, and the labels
		"""
		self.name = None
		self.constants = constants
		self.model = None
		self.history = None

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
  
		model = self.model
  
		# Initialize GridSearchCV
		grid_search = GridSearchCV(estimator=self.model, param_grid=hyperparameters_choices, cv=cv, scoring='accuracy')
		print(f"The GridSearchCV will test {grid_search.n_splits_} combinations of hyperparameters.")
		
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

	def get_alea_hyperparameters(self, X, Y, hyperparameters_choices):
		""" Return random combination of hyperparameters
		Paremeters:
			- X : np.array, shape (n_samples, n_features), training features.
			- Y : np.array, shape (n_samples,), training labels.
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
		self.history = self.model.fit(X, Y)

	def predict(self, X):
		""" Predict the labels of the samples in X.
		Parameters:
			- X : np.array, shape (n_samples, n_features)
		Output:
			- Y : np.array, shape (n_samples,)
		"""
		Y = self.model.predict(X)
		return Y

	def show_history(self):
		""" Show the history of the training """
		if self.history is None:
			print("No history found !")
		else:
			print("===========================================")
			print(f"History of the training of the {self.name} model :")
			print(f"Number of epochs : {len(self.history.history['loss'])}")
			print(f"Final loss       : {self.history.history['loss'][-1]}")
			print(f"Final accuracy   : {self.history.history['accuracy'][-1]}")
			print("===========================================")
			print("Plotting the history ...")
			self.plot_history()
	
	def plot_history(self):
		""" Plot the history of the training """
		if self.history is None:
			print("No history found !")
		else:
			import matplotlib.pyplot as plt
			plt.plot(self.history.history['loss'], label='loss')
			plt.plot(self.history.history['accuracy'], label='accuracy')
			plt.xlabel('Epoch')
			plt.legend()
			plt.show()
    
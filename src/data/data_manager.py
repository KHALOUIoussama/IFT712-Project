from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from src.data.constants import Constants
import os
from pretreatment import load_raw_data


class DataManager:
	def __init__(self, data_path, name, split_ratio=0.2):
		self.data_path = data_path
		self.split_ratio = split_ratio
		self.constants = None
		self.x_train = None
		self.t_train = None
		self.x_test = None
		self.t_test = None
		self.name = name

		self.load_data()

	def load_data(self):
		""" load data
		Output : 
		- constants : Constants, contains the number of labels, features and samples, and the labels
		- x_train : np.array, shape (n_samples, n_features)
		- x_test : np.array, shape (n_samples, n_features)
		- t_train : np.array, shape (n_samples, ), contains the labels
		- t_test : np.array, shape (n_samples, ), contains the labels
		"""
		self.train_data = load_raw_data(self.data_path)

		# duplicate the data 
		train_data = self.train_data.copy()

		# Encode the species column using LabelEncoder
		label_encoder = LabelEncoder()
		train_data['species'] = label_encoder.fit_transform(train_data['species'])

		# Extract features and labels
		X = train_data.drop(['id', 'species'], axis=1)
		Y = train_data['species']

		# Transform the value of Y into one-hot encoding
		# Y = np.eye(len(label_encoder.classes_))[Y]

		# Stratified Shuffle Split
		sss = StratifiedShuffleSplit(n_splits=1, test_size=self.split_ratio, random_state=42)

		for train_index, test_index in sss.split(X, Y):
			self.x_train, self.x_test = X.iloc[train_index], X.iloc[test_index]
			self.t_train, self.t_test = Y.iloc[train_index], Y.iloc[test_index]

		# Get the number of labels, features and samples, and the labels
		n_labels = len(label_encoder.classes_)
		n_features = X.shape[1]
		n_samples = X.shape[0]
		labels = label_encoder.classes_

		# Create a Constants object
		self.constants = Constants(n_labels, n_features, n_samples, labels)


	def save_test_data(self, Y_pred, path):
		"""
		Save the test data in a csv file.
		Parameters:
			- Y_pred : np.array, shape (n_samples, n_labels), contains the probabilities of each label
			- path : str, path to the csv file with the id and the probabilities of each label
		"""

		# Get the labels
		labels = self.constants.get_labels()

		# Create a DataFrame with the predicted labels
		df = pd.DataFrame(Y_pred, columns=labels)

		# Add the id column
		df.insert(0, 'id', self.test_data['id'])

		# Save the DataFrame to a CSV file
		df.to_csv(path, index=False)

	
	def one_hot_encoding(self, Y):
		""" Transform the value of Y into one-hot encoding
		Parameters:
			- Y : np.array, shape (n_samples, ), contains the labels
		Output:
			- Y : np.array, shape (n_samples, n_labels), one-hot encoding
		"""
		return np.eye(self.constants.get_n_labels())[Y]
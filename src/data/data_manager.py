from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from src.data.constants import Constants
import os
from pretreatment import load_raw_data, save_raw_data


class DataManager:
	def __init__(self, split_ratio=0.2, data_path="../data/raw/train.csv", test_path="../data/raw/test.csv"):
		self.data_path = data_path
		self.test_path = test_path
		self.split_ratio = split_ratio
		self.constants = None
		self.train_data = None
		self.test_data = None

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
			X_train, X_test = X.iloc[train_index], X.iloc[test_index]
			Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

		# Get the number of labels, features and samples, and the labels
		n_labels = len(label_encoder.classes_)
		n_features = X.shape[1]
		n_samples = X.shape[0]
		labels = label_encoder.classes_

		# Create a Constants object
		self.constants = Constants(n_labels, n_features, n_samples, labels)

		return self.constants, X_train, X_test, Y_train, Y_test

	def load_test_data(self):
		"""
		Load the test data. There is no label.
		Output:
			- constants : Constants, contains the number of labels, features and samples, and the labels
			- X_test : np.array, shape (n_samples, n_features)
		"""
		self.test_data = load_raw_data(self.test_path)

		# duplicate the data
		test_data = self.test_data.copy()

		# Extract features
		X_test = test_data.drop(['id'], axis=1)

		# Get the number of labels, features and samples, and the labels
		n_labels = self.constants.get_n_labels()
		n_features = X_test.shape[1]
		n_samples = X_test.shape[0]
		labels = self.constants.get_labels()

		# Create a Constants object
		constants = Constants(n_labels, n_features, n_samples, labels)

		return constants, X_test

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
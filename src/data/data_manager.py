from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from src.data.constants import Constants
import os
from pretreatment import load_raw_data, save_raw_data


class DataManager:
	def __init__(self, split_ratio=0.2, path="../data/raw/train.csv"):
		self.path = path
		self.split_ratio = split_ratio
		self.constants = None

	def load_data(self, super_classes=False):
		""" load data
		Parameters :
		- super_classes : bool, whether or not to create super classes using classes prefix
		Output : 
		- constants : Constants, contains the number of labels, features and samples, and the labels
		- x_train : np.array, shape (n_samples, n_features)
		- x_test : np.array, shape (n_samples, n_features)
		- t_train : np.array, shape (n_samples, ), contains the labels
		- t_test : np.array, shape (n_samples, ), contains the labels
		"""
		train_data = load_raw_data(self.path)

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
	
	def one_hot_encoding(self, Y):
		""" Transform the value of Y into one-hot encoding
		Parameters:
			- Y : np.array, shape (n_samples, ), contains the labels
		Output:
			- Y : np.array, shape (n_samples, n_labels), one-hot encoding
		"""
		return np.eye(self.constants.get_n_labels())[Y]
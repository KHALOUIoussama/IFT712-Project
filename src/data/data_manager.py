from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.data.constants import Constants
import os


class DataManager:
	def __init__(self, split_ratio=0.2):
		self.path = "../data/raw/train.csv"
		self.split_ratio = split_ratio
		self.constants = None

	def load_data(self):
		""" load data
		Output : 
		- constants : Constants, contains the number of labels, features and samples, and the labels
		- x_train : np.array, shape (n_samples, n_features)
		- x_test : np.array, shape (n_samples, n_features)
		- t_train : np.array, shape (n_samples, ), contains the labels
		- t_test : np.array, shape (n_samples, ), contains the labels
		"""
		if not os.path.isfile(self.path):
			print(f"File {self.path} not found !")
			return

		# Load the train.csv file
		train_data = pd.read_csv(self.path)

		# Encode the species column using LabelEncoder
		label_encoder = LabelEncoder()
		train_data['species'] = label_encoder.fit_transform(train_data['species'])

		# Extract features and labels
		X = train_data.drop(['id', 'species'], axis=1)
		Y = train_data['species']

		# Transform the value of Y into one-hot encoding
		# Y = np.eye(len(label_encoder.classes_))[Y]

		# Split the data into training and testing sets
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.split_ratio)

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
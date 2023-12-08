"""
This file contains the functions used to pretreat the data.
"""

import pandas as pd
import numpy as np
import os

def load_raw_data(path):
	"""
	Load the data from a csv file.

	Parameters:
		- path : str, path to the csv file

	Output:
		- data : pd.DataFrame, contains the data
	"""

	# Check if the file exists
	if not os.path.isfile(path):
		print(f"File {path} not found !")
		return
	
	# Load the csv file
	data = pd.read_csv(path)

	return data


def save_raw_data(data, path):
	"""
	Save the data in a csv file.

	Parameters:
		- data : pd.DataFrame, contains the data
		- path : str, path to the csv file
	"""

	# Save the data in a csv file
	data.to_csv(path, index=False)


def create_superclasses(normal_path, super_path):
	"""
	Create superclasses using the classes prefix.

	Parameters:
		- normal_path : str, path to the csv file containing the normal classes
		- super_path : str, path to the csv file containing the superclasses
	"""

	# Load the csv file containing the normal classes
	normal_data = load_raw_data(normal_path)

	# Create superclasses using the classes prefix
	normal_data['species'] = normal_data['species'].apply(lambda x: x.split('_')[0])

	# Save the data in a csv file
	save_raw_data(normal_data, super_path)
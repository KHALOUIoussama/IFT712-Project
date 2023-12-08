

class Constants:
	""" Contains the number of labels, features and samples, and the labels """

	def __init__(self, n_labels, n_features, n_samples, labels):
		self._n_labels = n_labels  	    # number of labels (species)
		self._n_features = n_features	# number of features (caracteristics of the leafs)
		self._n_samples = n_samples		# number of samples (leafs)
		self._labels = labels			# np.array, shape (n_labels, ), contains the labels
	
	def get_n_labels(self):
		return self._n_labels
	
	def get_n_features(self):
		return self._n_features
	
	def get_n_samples(self):
		return self._n_samples
	
	def get_labels(self):
		return self._labels

	
	def print(self):
		print("===========================================")
		print(f"n_labels   = {self._n_labels}")
		print(f"n_features = {self._n_features}")
		print(f"n_samples  = {self._n_samples}")
		print(f"labels     = {self._labels}")
		print("===========================================")

	
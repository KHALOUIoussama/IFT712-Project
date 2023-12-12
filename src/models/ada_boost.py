from model import Model
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost(Model):

	def __init__(self):
		self.name = "AdaBoost"
		self.model = AdaBoostClassifier()  # Initialize the model attribute with the AdaBoostClassifier

	def get_hyperparameters_choices(self):
		"""
		Return the hyperparameters to test for the model.
		"""
		return {
			"n_estimators": [262, 275, 287],
			"learning_rate": [0.12, 0.025, 0.037],
			'algorithm': ['SAMME', 'SAMME.R']
		}

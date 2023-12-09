from model import Model
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost(Model):

	def __init__(self, constants):
		super().__init__(constants)
		self.name = "AdaBoost"
		self.model = AdaBoostClassifier()  # Initialize the model attribute with the AdaBoostClassifier

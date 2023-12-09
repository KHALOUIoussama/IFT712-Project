from model import Model
import numpy as np


class AdaBoost(Model):

	def __init__(self, constants):
		super().__init__(constants)
		self.name = "AdaBoost"

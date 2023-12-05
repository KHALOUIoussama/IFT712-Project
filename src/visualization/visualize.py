import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd


class Visualize:
	def __init__(self, constants, t_test, t_pred):
		"""
		Parameters:
			- constants : Constants, contains the number of labels, features and samples, and the labels
			- t_test : np.array, shape (n_samples, n_labels), one-hot encoding
			- t_pred : np.array, shape (n_samples, n_labels), one-hot encoding
		"""
		self.constants = constants
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1_score = None
		self.mean_precision = None
		self.mean_recall = None
		self.mean_f1_score = None
		self.t_test = t_test
		self.t_pred = t_pred
		self.compute_scores()

	def compute_scores(self):
		"""
		This function analyses the results of the prediction. It computes the accuracy and the precision, the recall and the F1 score for each label.
		"""
		# Compute the accuracy
		self.accuracy = np.sum(np.argmax(self.t_test, axis=1) == np.argmax(self.t_pred, axis=1)) / self.t_test.shape[0]

		# Compute the precision, recall and F1 score
		self.precision = []
		self.recall = []
		self.f1_score = []
		for i in range(self.t_test.shape[1]):
			TP = np.sum((np.argmax(self.t_test, axis=1) == i) & (np.argmax(self.t_pred, axis=1) == i))
			FP = np.sum((np.argmax(self.t_test, axis=1) != i) & (np.argmax(self.t_pred, axis=1) == i))
			FN = np.sum((np.argmax(self.t_test, axis=1) == i) & (np.argmax(self.t_pred, axis=1) != i))
			self.precision.append(TP / (TP + FP))
			self.recall.append(TP / (TP + FN))
			self.f1_score.append(2 * self.precision[i] * self.recall[i] / (self.precision[i] + self.recall[i]))

		# Compute the mean precision, recall and F1 score
		self.mean_precision = np.mean(self.precision)
		self.mean_recall = np.mean(self.recall)
		self.mean_f1_score = np.mean(self.f1_score)

	def print_mean_scores(self):
		""" Print the mean scores """
		print("===========================================")
		print(f"Accuracy       = {self.accuracy}")
		print(f"Mean precision = {self.mean_precision}")
		print(f"Mean recall    = {self.mean_recall}")
		print(f"Mean F1 score  = {self.mean_f1_score}")
		print("===========================================")

	def print_labels_scores(self):
		""" Print the scores for each label """
		print("===========================================")
		print("| Label | Precision | Recall | F1 Score |")
		print("|-------|-----------|--------|----------|")
		for i in range(self.constants.get_n_labels()):
			label_name = self.constants.labels[i]
			print(f"| {i:<5} | {self.precision[i]:<9.4f} | {self.recall[i]:<6.4f} | {self.f1_score[i]:<8.4f} |")
		print("===========================================")

	def show_confusion_matrix(self):
		""" Show the confusion matrix """

		# Compute the confusion matrix
		cm = confusion_matrix(np.argmax(self.t_test, axis=1), np.argmax(self.t_pred, axis=1))

		# Plot the confusion matrix
		plt.figure(figsize=(10, 10))
		plt.imshow(cm, cmap=plt.cm.Blues)
		plt.title("Confusion matrix")
		plt.colorbar()
		tick_marks = np.arange(self.constants.get_n_labels())
		plt.xticks(tick_marks, self.constants.labels, rotation=90)
		plt.yticks(tick_marks, self.constants.labels)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()
		
		

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, log_loss
from sys import exit
import pandas as pd


class Visualize:
	def __init__(self, constants, t_test, t_pred, t_pred_proba):
		"""
		Parameters:
			- constants : Constants, contains the number of labels, features and samples, and the labels
			- t_test : np.array, shape (n_samples, n_labels)
			- t_pred : np.array, shape (n_samples, n_labels)
			- t_pred_proba : np.array, shape (n_samples, n_labels)
		"""
		self.constants = constants
		self.accuracy = None
		self.log_loss = 0
		self.precision = None
		self.recall = None
		self.f1_score = None
		self.mean_precision = None
		self.mean_recall = None
		self.mean_f1_score = None
		self.t_test = t_test
		self.t_pred = t_pred
		self.t_pred_proba = t_pred_proba

		self.compute_scores()

	def compute_scores(self):
		"""
		This function analyses the results of the prediction. It computes the accuracy and the precision, the recall and the F1 score for each label.
		"""
		# Compute the accuracy
		self.accuracy = np.sum(self.t_test == self.t_pred) / self.t_test.shape[0]

		# Compute the precision, recall and F1 score
		self.precision = []
		self.recall = []
		self.f1_score = []
		for i in range(self.constants.get_n_labels()):
			TP = np.sum((self.t_test == i) & (self.t_pred == i))
			FP = np.sum((self.t_test != i) & (self.t_pred == i))
			FN = np.sum((self.t_test == i) & (self.t_pred != i))

			if TP + FP == 0:
				precision = 0
			else:
				precision = TP / (TP + FP)

			if TP + FN == 0:
				recall = 0
			else:
				recall = TP / (TP + FN)

			if precision + recall == 0:
				f1_score = 0
			else:
				f1_score = 2 * precision * recall / (precision + recall)

			self.precision.append(precision)
			self.recall.append(recall)
			self.f1_score.append(f1_score)

		# Compute the mean precision, recall and F1 score
		self.mean_precision = np.mean(self.precision)
		self.mean_recall = np.mean(self.recall)
		self.mean_f1_score = np.mean(self.f1_score)
		if self.t_pred_proba is not None:
			self.log_loss = log_loss(self.t_test, self.t_pred_proba, labels=[i for i in range(self.constants.get_n_labels())])
    

	def print_mean_scores(self):
		""" Print the mean scores """
		print("===========================================")
		print(f"Log Loss       = {self.log_loss:.3f}")
		print(f"Accuracy       = {self.accuracy:.3f}")
		print(f"Mean precision = {self.mean_precision:.3f}")
		print(f"Mean recall    = {self.mean_recall:.3f}")
		print(f"Mean F1 score  = {self.mean_f1_score:.3f}")
		print("===========================================")

	def print_labels_scores(self):
		""" Print the scores for each label """
		print("===========================================")
		print("|         Label        | Precision | Recall | F1 Score |")
		print("|----------------------|-----------|--------|----------|")
		for i in range(self.constants.get_n_labels()):
			label_name = self.constants.get_labels()[i]
			if len(label_name) < 20:
				label_name += " " * (20 - len(label_name))
			else:
				label_name = label_name[:17] + "..."
			print(f"| {label_name} | {self.precision[i]:<9.3f} | {self.recall[i]:<6.3f} | {self.f1_score[i]:<8.3f} |")
		print("===========================================")
		print("| Mean                 | Precision | Recall | F1 Score |")
		print("|----------------------|-----------|--------|----------|")
		print(f"|                      | {self.mean_precision:<9.3f} | {self.mean_recall:<6.3f} | {self.mean_f1_score:<8.3f} |")


	def plot_confusion_matrix(self, show_label=False):
		""" Show the confusion matrix """

		# Compute the confusion matrix
		cm = confusion_matrix(self.t_test, self.t_pred)

		# Plot the confusion matrix

		if show_label:
			plt.figure(figsize=(10, 10))
		else:
			plt.figure(figsize=(5, 5))
		plt.imshow(cm, cmap=plt.cm.Blues)
		plt.title("Confusion matrix")
		plt.colorbar()
		tick_marks = np.arange(self.constants.get_n_labels())
		if show_label:
			plt.xticks(tick_marks, self.constants.get_labels(), rotation=90)
			plt.yticks(tick_marks, self.constants.get_labels())
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()

		

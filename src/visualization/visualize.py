import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import learning_curve
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
		self.accuracy = np.sum(self.t_test == self.t_pred) / self.t_test.shape[0] * 100

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
				precision = TP / (TP + FP) * 100

			if TP + FN == 0:
				recall = 0
			else:
				recall = TP / (TP + FN) * 100

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


	def plot_mean_scores(self, other_visualize=None, title='Model Performance Comparison'):
		if other_visualize is None:
			labels = ['Log Loss', 'Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1 Score']
			scores = [self.log_loss, self.accuracy, self.mean_precision, self.mean_recall, self.mean_f1_score]

			plt.figure(figsize=(8, 4))
			plt.bar(labels, scores, color=['orange', 'blue', 'green', 'red', 'purple'])
			plt.xlabel('Metrics')
			plt.ylabel('Percentage')
			plt.title(title)
			plt.ylim(0, 100)
			for i, score in enumerate(scores):
				plt.text(i, score + 2, f'{score:.2f}', ha='center')
			plt.show()
		else:
			labels = ['Log Loss', 'Accuracy', 'Mean Precision', 'Mean Recall', 'Mean F1 Score']
			current_scores = [self.log_loss, self.accuracy, self.mean_precision, self.mean_recall, self.mean_f1_score]

			x = np.arange(len(labels))  # Les labels de l'axe x
			width = 0.35  # La largeur des barres

			fig, ax = plt.subplots()
			rects1 = ax.bar(x - width/2, current_scores, width, label='Current Model')

			if other_visualize:
				other_scores = [other_visualize.log_loss, other_visualize.accuracy, other_visualize.mean_precision, other_visualize.mean_recall, other_visualize.mean_f1_score]
				rects2 = ax.bar(x + width/2, other_scores, width, label='Optimized Model')

			ax.set_ylabel('Scores')
			ax.set_title(title)
			ax.set_xticks(x)
			ax.set_xticklabels(labels)
			ax.legend()

			plt.show()



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
			print(f"| {label_name} | {self.precision[i]:<9.3f}% | {self.recall[i]:<6.3f}% | {self.f1_score[i]:<8.3f}% |")
		print("===========================================")
		print("| Mean                 | Precision | Recall | F1 Score |")
		print("|----------------------|-----------|--------|----------|")
		print(f"|                      | {self.mean_precision:<9.3f}% | {self.mean_recall:<6.3f}% | {self.mean_f1_score:<8.3f}% |")


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

	def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
		"""
		Génère une courbe d'apprentissage simple.
		"""
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score")

		plt.legend(loc="best")
		return plt
		

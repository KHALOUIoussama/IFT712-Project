from sklearn.svm import SVC
from model import Model
from sklearn.model_selection import GridSearchCV

class Svm(Model):
    def __init__(self):
        self.name = "SVM"
        self.model = SVC()

    def get_hyperparameters_choices(self):
        """
        Return the hyperparameters to test for the model.
        """
        return {
            "C": [1, 10, 100],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['scale', 'auto']
        }

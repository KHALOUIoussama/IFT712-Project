from model import Model
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Model):

    def __init__(self):
        self.name = "DecisionTree"
        self.model = DecisionTreeClassifier()  # Initialisation du modèle DecisionTreeClassifier

    def get_hyperparameters_choices(self):
        """
        Return the hyperparameters to test for the model.
        """
        return {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': ['sqrt', 'log2', None],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }
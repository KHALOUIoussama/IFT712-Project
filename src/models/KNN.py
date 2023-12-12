from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from model import Model

class KNN(Model):
    def __init__(self):
        self.name = "KNN"

        self.model = SklearnKNN()

    def get_hyperparameters_choices(self):
        """
        Return the hyperparameters to test for the model.
        """
        return {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

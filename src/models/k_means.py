from sklearn.cluster import KMeans as SklearnKMeans
from model import Model

class KMeans(Model):
    def __init__(self, n_clusters):
        self.name = "KMeans"

        self.model = SklearnKMeans(n_clusters=n_clusters)

    def get_hyperparameters_choices(self):
        """
        Return the hyperparameters to test for the model.
        """
        return {
            'n_init': ["auto", 10, 30, 50],
            'algorithm' : ['auto', 'full', 'elkan', 'lloyd']
        }

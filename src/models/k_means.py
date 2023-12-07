from sklearn.cluster import KMeans
from model import Model

class KMeansModel(Model):

    def __init__(self, constants):
        super().__init__(constants)
        self.name = "KMeans"
        self.model = None

    def find_optimal_hyperparameters(self, X, hyperparameters, cv=5):
        # K-Means doesn't have hyperparameters to tune
        pass

    def train(self, X, hyperparameters):
        # Train the K-Means model with the specified hyperparameters
        self.model = KMeans(n_clusters=hyperparameters.get('n_clusters', 8))
        self.model.fit(X)

    def predict(self, X):
        # Predict the cluster labels for input data X
        if not self.model:
            raise ValueError("The K-Means model has not been trained yet. Please call the train method.")

        return self.model.predict(X)

from sklearn.cluster import KMeans
from model import Model
from sklearn.model_selection import GridSearchCV

class KMeans(Model):
    def __init__(self, constants):
        super().__init__(constants)
        self.name = "KMeans"
        self.model = None

    def find_optimal_hyperparameters(self, X, Y, hyperparameters, cv=5):
        # KMeans doesn't have hyperparameters like SVM, but we can optimize the number of clusters (k)
        param_grid = {'n_clusters': hyperparameters.get('n_clusters', [2, 3, 4, 5])}
        grid_search = GridSearchCV(estimator=KMeans(), param_grid=param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X, Y)
        best_hyperparameters = grid_search.best_params_
        return best_hyperparameters

    def train(self, X, Y, hyperparameters):
        # Train the KMeans model with the specified number of clusters
        self.model = KMeans(n_clusters=hyperparameters.get('n_clusters', 2))
        self.model.fit(X)

    def predict(self, X):
        # Predict the labels for input data X
        if not self.model:
            raise ValueError("The KMeans model has not been trained yet. Please call the train method.")
        return self.model.predict(X)

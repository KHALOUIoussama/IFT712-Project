from sklearn.svm import SVC
from model import Model
from sklearn.model_selection import GridSearchCV

class Svm(Model):
    def __init__(self, constants):
        super().__init__(constants)
        self.name = "SVM"
        self.model = None

    def find_optimal_hyperparameters(self, X, Y, hyperparameters, cv=5):
        # SVM doesn't have hyperparameters in the traditional sense, but we can optimize C (regularization parameter)
        param_grid = {'C': hyperparameters.get('C', [1, 10, 100, 1000])}
        grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X, Y)
        best_hyperparameters = grid_search.best_params_
        return best_hyperparameters

    def train(self, X, Y, hyperparameters):
        # Train the SVM model with the specified hyperparameters
        self.model = SVC(C=hyperparameters.get('C', 1))
        self.model.fit(X, Y)

    def predict(self, X):
        # Predict the labels for input data X
        if not self.model:
            raise ValueError("The SVM model has not been trained yet. Please call the train method.")
        return self.model.predict(X)


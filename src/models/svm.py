from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from model import Model

class Svm(Model):

    def __init__(self, constants):
        super().__init__(constants)
        self.name = "Svm"
        self.model = None

    def find_optimal_hyperparameters(self, X, Y, hyperparameters, cv=5):
        # Specify the parameter grid for hyperparameter tuning
        param_grid = {
            'C': hyperparameters.get('C', [1.0]),
            'kernel': hyperparameters.get('kernel', ['rbf']),
            'gamma': hyperparameters.get('gamma', ['scale']),
        }

        # Create the SVM model
        svm_model = SVC()

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(svm_model, param_grid, cv=cv)
        grid_search.fit(X, Y)

        # Set the optimal hyperparameters to the model
        self.model = grid_search.best_estimator_

        # Print the optimal hyperparameters
        print("Optimal Hyperparameters:", grid_search.best_params_)

    def train(self, X, Y, hyperparameters):
        # Train the SVM model with the specified hyperparameters
        if not self.model:
            self.find_optimal_hyperparameters(X, Y, hyperparameters)

        # Fit the model to the training data
        self.model.fit(X, Y)

    def predict(self, X):
        # Predict the class labels for input data X
        if not self.model:
            raise ValueError("The SVM model has not been trained yet. Please call the train method.")

        return self.model.predict(X)

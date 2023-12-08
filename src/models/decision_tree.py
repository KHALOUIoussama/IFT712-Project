from model import Model
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Model):

    def __init__(self, constants):
        super().__init__(constants)
        self.name = "DecisionTree"
        self.model = DecisionTreeClassifier()  # Initialisation du modèle DecisionTreeClassifier

    def find_optimal_hyperparameters(self, X, Y, hyperparameters, cv=5):
        # Utilisez la méthode de la classe parente car elle est déjà adéquate
        return super().find_optimal_hyperparameters(X, Y, hyperparameters, cv)

    def train(self, X, Y, hyperparameters):
        # Mettez à jour le modèle avec les hyperparamètres optimaux
        self.model.set_params(**hyperparameters)
        # Entraînement du modèle
        self.model.fit(X, Y)

    def predict(self, X):
        # Utiliser le modèle entraîné pour prédire
        return self.model.predict(X)

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_

    def get_name(self):
        return "Logistic Regression"

    def visualize(self, X, y, save_path="logistic_regression_plot.png"):
        if X.shape[1] != 1:
            print("⚠️ Visualization only works for 1D features")
            return None

        X_vals = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        probs = self.model.predict_proba(X_vals)[:, 1]

        plt.figure(figsize=(6, 4))
        plt.scatter(X, y, color='blue', label='Actual')
        plt.plot(X_vals, probs, color='red', linewidth=2, label='Prediction Probabilities')
        plt.title('Logistic Regression')
        plt.xlabel('X')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

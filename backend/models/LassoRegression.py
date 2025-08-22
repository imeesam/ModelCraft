from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

class LasooRegressionModel:
    def __init__(self, alpha=1.0):
        self.model = Lasso(alpha=alpha)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_

    def get_name(self):
        return "Lasso Regression"

    def visualize(self, X, y, save_path="lasso_regression_plot.png"):
        y_pred = self.predict(X)
        plt.figure(figsize=(6, 4))
        plt.scatter(X, y, color='blue', label='Actual')
        plt.plot(X, y_pred, color='green', linewidth=2, label='Predicted')
        plt.title('Lasso Regression')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

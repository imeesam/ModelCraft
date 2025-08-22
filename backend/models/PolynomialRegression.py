from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class Polyregression:
    def __init__(self,poly):
        self.poly = PolynomialFeatures(degree=poly)
        self.model = LinearRegression()

    def get_name(self):
        return "Polynomial Regression"

    def train(self,X,y):
        X_poly = self.poly.fit_transform(X)
        self.poly.fit(X_poly,y)
        self.model.fit(X_poly,y)

    def predict(self,X):
        return self.model.predict(self.poly.transform(X))
    
    def evaluate(self,X,y):
        X_poly = self.poly.transform(X)
        return self.model.score(X_poly,y)

    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_
    
    def visualize(self, X, y, save_path="poly_regression_plot.png"):
        y_pred = self.predict(X)
        plt.figure(figsize=(6,4))
        plt.scatter(X, y, color='blue', label='Actual')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
        plt.title('Polynomial Regression')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

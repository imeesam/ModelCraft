from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class linearregression:
    def __init__(self):
        self.model = LinearRegression()

    def train(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)
    
    def evaluate(self,X,y):
        return self.model.score(X,y)

    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_
    
    def get_name(self):
        return "Linear Regression"
    
    def visualize(self, X, y, save_path="linear_regression_plot.png"):
        y_pred = self.predict(X)
        plt.figure(figsize=(6, 4))
        plt.scatter(X, y, color='blue', label='Actual')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
        plt.title('Linear Regression')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

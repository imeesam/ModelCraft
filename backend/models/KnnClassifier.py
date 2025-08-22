from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.n_neighbors = n_neighbors

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def get_name(self):
        return f"k-Nearest Neighbors (k={self.n_neighbors})"

    def visualize(self, X, y, save_path="knn_plot.png"):
        # Works only for 2D feature input
        h = .02  # step size in the mesh

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
        plt.title(f"Decision Boundary for KNN (k={self.n_neighbors})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

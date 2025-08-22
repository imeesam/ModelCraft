from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansClusteringModel:
    def __init__(self, n_clusters=3, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.n_clusters = n_clusters

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X):
        # In unsupervised learning, inertia is commonly used
        return self.model.inertia_

    def get_name(self):
        return f"K-Means Clustering (k={self.n_clusters})"

    def visualize(self, X, save_path="kmeans_plot.png"):
        y_kmeans = self.model.predict(X)
        centers = self.model.cluster_centers_

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
        plt.title("K-Means Clustering")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

class DecisionTreeClassifierModel:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def get_name(self):
        return "Decision Tree Classifier"

    def visualize(self, feature_names=None, class_names=None, save_path="decision_tree_plot.png"):
        plt.figure(figsize=(12, 8))
        tree.plot_tree(self.model, filled=True, feature_names=feature_names, class_names=class_names)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

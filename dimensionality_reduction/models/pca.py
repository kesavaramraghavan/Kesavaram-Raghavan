import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness


class PCAReducer:
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = PCA(n_components=n_components, random_state=random_state)
        self.model_name = "PCA"

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def evaluate(self, X, embedding):
        metrics = {
            'trustworthiness': trustworthiness(X, embedding, n_neighbors=5),
            'explained_variance': float(np.sum(self.model.explained_variance_ratio_)),
        }
        return metrics



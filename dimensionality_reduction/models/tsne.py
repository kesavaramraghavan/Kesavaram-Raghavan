import numpy as np
from sklearn.manifold import TSNE, trustworthiness


class TSNEReducer:
    def __init__(self, n_components=2, perplexity=30.0, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init='pca')
        self.model_name = "t-SNE"

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def evaluate(self, X, embedding):
        metrics = {
            'trustworthiness': trustworthiness(X, embedding, n_neighbors=5),
            'explained_variance': None,
        }
        return metrics



import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score


class GMMClusteringModel:
    def __init__(self, k_values=None, covariance_type='full', random_state=42):
        self.k_values = k_values or [2, 3, 4, 5, 6]
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.best_model = None
        self.best_k = None
        self.model_name = f"GMM ({covariance_type})"

    def find_best_parameters(self, X, y_true=None):
        best_score = -np.inf
        for k in self.k_values:
            model = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=self.random_state)
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels)
            if sil > best_score:
                best_score = sil
                self.best_k = k
        return self.best_k

    def fit(self, X):
        if self.best_k is None:
            self.find_best_parameters(X)
        self.best_model = GaussianMixture(n_components=self.best_k, covariance_type=self.covariance_type, random_state=self.random_state)
        self.best_model.fit(X)
        return self

    def predict(self, X):
        return self.best_model.predict(X)

    def evaluate(self, X, y_true=None):
        labels = self.predict(X)
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'k': self.best_k,
        }
        if y_true is not None:
            metrics['ari'] = adjusted_rand_score(y_true, labels)
            metrics['nmi'] = normalized_mutual_info_score(y_true, labels)
        return metrics



import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score


class DBSCANClusteringModel:
    def __init__(self, eps_values=None, min_samples_values=None):
        self.eps_values = eps_values or [0.2, 0.5, 0.8, 1.0]
        self.min_samples_values = min_samples_values or [3, 5, 10]
        self.best_model = None
        self.best_params = None
        self.model_name = "DBSCAN"

    def find_best_parameters(self, X, y_true=None):
        best_score = -np.inf
        for eps in self.eps_values:
            for ms in self.min_samples_values:
                model = DBSCAN(eps=eps, min_samples=ms)
                labels = model.fit_predict(X)
                # If all points are noise or one cluster, skip silhouette
                if len(set(labels)) <= 1 or (set(labels) == {-1}):
                    continue
                sil = silhouette_score(X, labels)
                if sil > best_score:
                    best_score = sil
                    self.best_params = {'eps': eps, 'min_samples': ms}
        return self.best_params

    def fit(self, X):
        if self.best_params is None:
            self.find_best_parameters(X)
        self.best_model = DBSCAN(**self.best_params)
        self.best_model.fit(X)
        return self

    def predict(self, X):
        # DBSCAN has no predict; re-fit and return labels
        model = DBSCAN(**self.best_params)
        return model.fit_predict(X)

    def evaluate(self, X, y_true=None):
        labels = self.predict(X)
        metrics = {
            'silhouette': silhouette_score(X, labels) if len(set(labels)) > 1 and -1 not in set(labels) else -1,
            'davies_bouldin': davies_bouldin_score(X, labels) if len(set(labels)) > 1 and -1 not in set(labels) else np.inf,
            'calinski_harabasz': calinski_harabasz_score(X, labels) if len(set(labels)) > 1 and -1 not in set(labels) else -1,
            'eps': self.best_params['eps'],
            'min_samples': self.best_params['min_samples'],
        }
        if y_true is not None:
            metrics['ari'] = adjusted_rand_score(y_true, labels)
            metrics['nmi'] = normalized_mutual_info_score(y_true, labels)
        return metrics



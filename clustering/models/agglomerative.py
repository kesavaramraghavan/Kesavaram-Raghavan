import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score


class AgglomerativeClusteringModel:
    def __init__(self, k_values=None, linkage='ward'):
        self.k_values = k_values or [2, 3, 4, 5, 6]
        self.linkage = linkage
        self.best_model = None
        self.best_k = None
        self.model_name = f"Agglomerative ({linkage})"

    def find_best_parameters(self, X, y_true=None):
        best_score = -np.inf
        for k in self.k_values:
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels)
            if sil > best_score:
                best_score = sil
                self.best_k = k
        return self.best_k

    def fit(self, X):
        if self.best_k is None:
            self.find_best_parameters(X)
        self.best_model = AgglomerativeClustering(n_clusters=self.best_k, linkage=self.linkage)
        self.best_model.fit(X)
        return self

    def predict(self, X):
        # AgglomerativeClustering does not have predict; re-fit on X and return labels
        model = AgglomerativeClustering(n_clusters=self.best_k, linkage=self.linkage)
        return model.fit_predict(X)

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



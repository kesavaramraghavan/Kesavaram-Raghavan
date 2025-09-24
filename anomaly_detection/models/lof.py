import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score


class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
        self.model_name = "Local Outlier Factor"

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        # Negative outlier factor: lower values indicate outliers; convert to positive anomaly score
        return -self.model.score_samples(X)

    def predict_labels(self, X, threshold=None):
        scores = self.score(X)
        if threshold is None:
            threshold = np.quantile(scores, 0.95)
        return (scores >= threshold).astype(int)

    def evaluate(self, X, y_true=None):
        scores = self.score(X)
        metrics = {
            'median_score': float(np.median(scores)),
            'mean_score': float(np.mean(scores)),
        }
        if y_true is not None and set(np.unique(y_true)) <= {0, 1}:
            metrics['roc_auc'] = roc_auc_score(y_true, scores)
        return metrics



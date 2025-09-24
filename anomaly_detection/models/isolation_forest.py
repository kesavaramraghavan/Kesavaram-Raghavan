import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score


class IsolationForestModel:
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.model_name = "Isolation Forest"

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        # Higher is more normal. Convert to anomaly score as negative
        return -self.model.score_samples(X)

    def predict_labels(self, X, threshold=None):
        scores = self.score(X)
        if threshold is None:
            # Use quantile based on contamination
            threshold = np.quantile(scores, 1 - (1 - self.contamination))
        return (scores >= threshold).astype(int)  # 1 for anomaly

    def evaluate(self, X, y_true=None):
        scores = self.score(X)
        metrics = {
            'median_score': float(np.median(scores)),
            'mean_score': float(np.mean(scores)),
        }
        if y_true is not None and set(np.unique(y_true)) <= {0, 1}:
            metrics['roc_auc'] = roc_auc_score(y_true, scores)
        return metrics



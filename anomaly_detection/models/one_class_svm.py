import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score


class OneClassSVMModel:
    def __init__(self, nu=0.05, kernel='rbf', gamma='scale'):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.model_name = f"One-Class SVM ({kernel})"

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        # Higher is more normal; make anomaly score by negating
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



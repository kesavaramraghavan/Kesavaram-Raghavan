import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score


class KNNClassifierModel:
    def __init__(self, k_values=None):
        self.k_values = k_values or [3, 5, 7, 11]
        self.best_model = None
        self.best_k = None
        self.model_name = "K-Nearest Neighbors"

    def find_best_parameters(self, X, y, cv=5):
        best_score = -np.inf
        for k in self.k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            if scores.mean() > best_score:
                best_score = scores.mean()
                self.best_k = k
        return self.best_k

    def fit(self, X, y):
        if self.best_k is None:
            self.find_best_parameters(X, y)
        self.best_model = KNeighborsClassifier(n_neighbors=self.best_k)
        self.best_model.fit(X, y)
        return self

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        return None

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'k': self.best_k,
        }
        if len(np.unique(y)) <= 2:
            proba = self.predict_proba(X)
            if proba is not None:
                metrics['roc_auc'] = roc_auc_score(y, proba[:, 1])
        return metrics

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.best_model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'cv_scores': scores,
        }

    def plot_confusion(self, X, y, labels=None):
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred, labels=labels)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues')
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.colorbar()
        plt.tight_layout()
        plt.show()



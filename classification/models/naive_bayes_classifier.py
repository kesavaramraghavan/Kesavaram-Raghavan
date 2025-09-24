import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score


class NaiveBayesClassifierModel:
    def __init__(self):
        self.best_model = GaussianNB()
        self.model_name = "Naive Bayes"

    def fit(self, X, y):
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



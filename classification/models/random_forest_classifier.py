import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score


class RandomForestClassifierModel:
    def __init__(self, n_estimators_range=None, max_depth_range=None):
        self.n_estimators_range = n_estimators_range or [50, 100, 200]
        self.max_depth_range = max_depth_range or [None, 5, 10, 15]
        self.best_model = None
        self.best_n_estimators = None
        self.best_max_depth = None
        self.model_name = "Random Forest (Classifier)"

    def find_best_parameters(self, X, y, cv=5):
        best_score = -np.inf
        for n in self.n_estimators_range:
            for d in self.max_depth_range:
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    self.best_n_estimators = n
                    self.best_max_depth = d
        return self.best_n_estimators, self.best_max_depth

    def fit(self, X, y):
        if self.best_n_estimators is None:
            self.find_best_parameters(X, y)
        self.best_model = RandomForestClassifier(
            n_estimators=self.best_n_estimators,
            max_depth=self.best_max_depth,
            random_state=42
        )
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
            'n_estimators': self.best_n_estimators,
            'max_depth': self.best_max_depth,
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



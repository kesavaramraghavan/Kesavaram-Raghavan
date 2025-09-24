#!/usr/bin/env python3
"""
Main Classification Analysis Script
Trains multiple classifiers and finds the best performing ones
"""

import argparse
import os
from typing import List
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from models.logistic_regression import LogisticRegressionClassifier
from models.svm_classifier import SVMClassifier
from models.random_forest_classifier import RandomForestClassifierModel
from models.xgboost_classifier import XGBoostClassifierModel
from models.knn_classifier import KNNClassifierModel
from models.naive_bayes_classifier import NaiveBayesClassifierModel
from models.decision_tree_classifier import DecisionTreeClassifierModel

warnings.filterwarnings('ignore')


class ClassificationAnalysis:
    def __init__(self, dataset_path: str, x_columns: List[str], y_column: str):
        self.dataset_path = dataset_path
        self.x_columns = x_columns
        self.y_column = y_column
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def _maybe_prepare_default_dataset(self):
        # If CSV does not exist, auto-create Iris dataset at dataset/iris.csv
        if not os.path.exists(self.dataset_path):
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df = iris.frame
            # Ensure target column name matches requested y_column
            if 'target' not in df.columns:
                df['target'] = iris.target
            df.to_csv(self.dataset_path, index=False)
            if not self.x_columns:
                self.x_columns = iris.feature_names
            if not self.y_column:
                self.y_column = 'target'

    def load_data(self):
        self._maybe_prepare_default_dataset()
        df = pd.read_csv(self.dataset_path)
        X = df[self.x_columns].values
        y = df[self.y_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.data = df
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        return self

    def initialize_models(self):
        self.models = {
            'Logistic Regression': LogisticRegressionClassifier(),
            'SVM (RBF)': SVMClassifier(kernel='rbf'),
            'Random Forest': RandomForestClassifierModel(),
            'XGBoost': XGBoostClassifierModel(),
            'KNN': KNNClassifierModel(),
            'Naive Bayes': NaiveBayesClassifierModel(),
            'Decision Tree': DecisionTreeClassifierModel(),
        }
        return self

    def train_models(self):
        for name, model in self.models.items():
            start = time.time()
            model.fit(self.X_train, self.y_train)
            test_metrics = model.evaluate(self.X_test, self.y_test)
            cv_results = model.cross_validate(self.X_train, self.y_train)
            self.results[name] = {
                'model': model,
                'test_metrics': test_metrics,
                'cv_results': cv_results,
                'training_time': time.time() - start,
            }
        return self

    def select_best_models(self, n_best=3):
        sorted_models = sorted(
            self.results.items(), key=lambda x: x[1]['test_metrics']['accuracy'], reverse=True
        )
        self.best_models = sorted_models[:n_best]
        return self

    def plot_model_comparison(self):
        names = list(self.results.keys())
        acc = [self.results[n]['test_metrics']['accuracy'] for n in names]
        f1w = [self.results[n]['test_metrics']['f1'] for n in names]
        cv = [self.results[n]['cv_results']['mean_cv_score'] for n in names]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.bar(names, acc, color='skyblue')
        ax1.set_title('Accuracy (Test)')
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax2.bar(names, f1w, color='lightgreen')
        ax2.set_title('F1-Weighted (Test)')
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax3.bar(names, cv, color='lightcoral')
        ax3.set_title('Accuracy (CV)')
        ax3.set_xticklabels(names, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        return self

    def _plot_confusion_matrices_for_best(self):
        if not hasattr(self, 'best_models') or not self.best_models:
            return
        n_best = len(self.best_models)
        fig, axes = plt.subplots(1, n_best, figsize=(6*n_best, 5))
        if n_best == 1:
            axes = [axes]
        labels = np.unique(self.y_test)
        for i, (name, result) in enumerate(self.best_models):
            model = result['model']
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred, labels=labels)
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'{name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels, rotation=0)
        plt.tight_layout()
        plt.show()

    def _plot_roc_curves_for_best(self):
        if not hasattr(self, 'best_models') or not self.best_models:
            return
        classes = np.unique(self.y_test)
        is_binary = len(classes) == 2
        # Binarize for multiclass
        if not is_binary:
            y_test_bin = label_binarize(self.y_test, classes=classes)
        plt.figure(figsize=(8, 6))
        for name, result in self.best_models:
            model = result['model']
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(self.X_test)
            if proba is None:
                # Skip ROC if no probabilities
                continue
            if is_binary:
                fpr, tpr, _ = roc_curve(self.y_test, proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
            else:
                # Macro-average ROC AUC
                y_test_bin = label_binarize(self.y_test, classes=classes)
                fpr_dict = {}
                tpr_dict = {}
                auc_list = []
                for i in range(len(classes)):
                    fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], proba[:, i])
                    fpr_dict[i], tpr_dict[i] = fpr_i, tpr_i
                    auc_list.append(auc(fpr_i, tpr_i))
                # Simple average of AUCs
                macro_auc = np.mean(auc_list)
                # Plot one vs rest for the class with highest AUC for clarity
                best_i = int(np.argmax(auc_list))
                plt.plot(fpr_dict[best_i], tpr_dict[best_i], lw=2, label=f'{name} (macro AUC={macro_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (Top Models)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def generate_report(self):
        print("\nClassification Report")
        print(f"Dataset: {self.dataset_path}")
        print(f"Features: {', '.join(self.x_columns)}")
        print(f"Target: {self.y_column}")
        print("\nModel Performance (sorted by accuracy):")
        sorted_models = sorted(
            self.results.items(), key=lambda x: x[1]['test_metrics']['accuracy'], reverse=True
        )
        for name, res in sorted_models:
            m = res['test_metrics']
            print(f"- {name}: acc={m['accuracy']:.4f}, f1={m['f1']:.4f}, prec={m['precision']:.4f}, recall={m['recall']:.4f}")
        if hasattr(self, 'best_models'):
            print("\nTop 3 Models:")
            for i, (name, res) in enumerate(self.best_models, 1):
                print(f"  {i}. {name} (acc={res['test_metrics']['accuracy']:.4f})")
        return self

    def run(self):
        (self.load_data()
         .initialize_models()
         .train_models()
         .select_best_models()
         .plot_model_comparison())
        # Detailed plots for best models
        self._plot_confusion_matrices_for_best()
        self._plot_roc_curves_for_best()
        self.generate_report()
        return True


def _infer_numeric_features(df: pd.DataFrame, target: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return [c for c in numeric_cols if c != target]


def main():
    parser = argparse.ArgumentParser(description="Train multiple classifiers and select the best ones")
    parser.add_argument("--dataset", type=str, default="dataset/iris.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name")
    parser.add_argument("--features", type=str, default="", help="Comma-separated feature columns; if omitted, auto-infer numeric features")
    args = parser.parse_args()

    if args.features.strip():
        x_cols = [c.strip() for c in args.features.split(',') if c.strip()]
    else:
        # If CSV missing, create default Iris
        if not os.path.exists(args.dataset):
            os.makedirs(os.path.dirname(args.dataset), exist_ok=True)
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            iris.frame.to_csv(args.dataset, index=False)
            df_preview = iris.frame
        else:
            df_preview = pd.read_csv(args.dataset)
        x_cols = _infer_numeric_features(df_preview, args.target) if not df_preview.empty else []

    analysis = ClassificationAnalysis(args.dataset, x_cols, args.target)
    analysis.run()


if __name__ == "__main__":
    main()



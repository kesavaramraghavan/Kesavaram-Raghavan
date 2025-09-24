#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.isolation_forest import IsolationForestModel
from models.one_class_svm import OneClassSVMModel
from models.lof import LOFModel


class AnomalyAnalysis:
    def __init__(self, dataset_path: str, x_columns: List[str], labels_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.x_columns = x_columns
        self.labels_path = labels_path
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def _prepare_default(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            iris.frame.to_csv(self.dataset_path, index=False)

    def load_data(self):
        self._prepare_default()
        df = pd.read_csv(self.dataset_path)
        if not self.x_columns:
            self.x_columns = df.select_dtypes(include=['number']).columns.tolist()
        X = df[self.x_columns].values
        self.X = self.scaler.fit_transform(X)
        self.labels = None
        if self.labels_path and os.path.exists(self.labels_path):
            self.labels = pd.read_csv(self.labels_path).values.squeeze()
        return self

    def initialize_models(self):
        self.models = {
            'Isolation Forest': IsolationForestModel(),
            'One-Class SVM': OneClassSVMModel(),
            'LOF': LOFModel(),
        }
        return self

    def run_models(self):
        for name, model in self.models.items():
            model.fit(self.X)
            metrics = model.evaluate(self.X, self.labels)
            self.results[name] = {'model': model, 'metrics': metrics}
        return self

    def plot_scores(self):
        # Histogram of anomaly scores for top 3 by median_score
        sorted_items = sorted(self.results.items(), key=lambda x: x[1]['metrics']['median_score'], reverse=True)
        top = sorted_items[:3]
        plt.figure(figsize=(12, 4))
        for i, (name, res) in enumerate(top, 1):
            plt.subplot(1, len(top), i)
            scores = res['model'].score(self.X)
            plt.hist(scores, bins=30, color='steelblue', alpha=0.8)
            plt.title(f"{name}")
            plt.xlabel('Anomaly score')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # PCA scatter colored by anomaly score for best model
        if top:
            name, res = top[0]
            scores = res['model'].score(self.X)
            pca = PCA(n_components=2)
            XY = pca.fit_transform(self.X)
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(XY[:, 0], XY[:, 1], c=scores, cmap='viridis', s=20)
            plt.title(f"PCA Scatter colored by anomaly score: {name}")
            plt.colorbar(sc, label='Anomaly score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return self

    def report(self):
        print("Anomaly Detection Report")
        for name, res in self.results.items():
            m = res['metrics']
            print(f"- {name}: mean_score={m['mean_score']:.3f}, median_score={m['median_score']:.3f}" + (f", roc_auc={m['roc_auc']:.3f}" if 'roc_auc' in m else ""))
        return self

    def run(self):
        (self.load_data()
         .initialize_models()
         .run_models()
         .plot_scores()
         .report())
        return True


def _infer_numeric_features(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['number']).columns.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection on a dataset")
    parser.add_argument("--dataset", type=str, default="dataset/iris.csv", help="Path to CSV with features")
    parser.add_argument("--features", type=str, default="", help="Comma-separated feature columns; auto-infer if omitted")
    parser.add_argument("--labels", type=str, default="", help="Optional CSV path with binary labels (0=normal,1=anomaly)")
    args = parser.parse_args()

    if args.features.strip():
        x_cols = [c.strip() for c in args.features.split(',') if c.strip()]
    else:
        df_preview = pd.read_csv(args.dataset) if os.path.exists(args.dataset) else pd.DataFrame()
        x_cols = _infer_numeric_features(df_preview) if not df_preview.empty else []

    analysis = AnomalyAnalysis(args.dataset, x_cols, args.labels or None)
    analysis.run()


if __name__ == "__main__":
    main()


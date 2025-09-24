#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples

from models.kmeans import KMeansClustering
from models.agglomerative import AgglomerativeClusteringModel
from models.dbscan import DBSCANClusteringModel
from models.gmm import GMMClusteringModel


class ClusteringAnalysis:
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
            iris.frame.drop(columns=['target'], inplace=True)
            iris.frame.to_csv(self.dataset_path, index=False)

    def load_data(self):
        self._prepare_default()
        df = pd.read_csv(self.dataset_path)
        if not self.x_columns:
            self.x_columns = df.select_dtypes(include=['number']).columns.tolist()
        X = df[self.x_columns].values
        self.X = self.scaler.fit_transform(X)
        self.df = df
        self.y_true = None
        if self.labels_path and os.path.exists(self.labels_path):
            y = pd.read_csv(self.labels_path).values.squeeze()
            self.y_true = y
        return self

    def initialize_models(self):
        self.models = {
            'KMeans': KMeansClustering(),
            'Agglomerative': AgglomerativeClusteringModel(),
            'DBSCAN': DBSCANClusteringModel(),
            'GMM': GMMClusteringModel(),
        }
        return self

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X)
            metrics = model.evaluate(self.X, self.y_true)
            self.results[name] = {'model': model, 'metrics': metrics}
        return self

    def select_best_models(self, n_best=3):
        sorted_items = sorted(self.results.items(), key=lambda x: x[1]['metrics']['silhouette'], reverse=True)
        self.best_models = sorted_items[:n_best]
        return self

    def _plot_pca_scatter(self, labels, ax, title):
        pca = PCA(n_components=2)
        XY = pca.fit_transform(self.X)
        scatter = ax.scatter(XY[:, 0], XY[:, 1], c=labels, cmap='tab10', s=25)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def plot_best_clusters(self):
        n_best = len(self.best_models)
        fig, axes = plt.subplots(1, n_best, figsize=(6*n_best, 5))
        if n_best == 1:
            axes = [axes]
        for i, (name, res) in enumerate(self.best_models):
            model = res['model']
            labels = model.predict(self.X)
            self._plot_pca_scatter(labels, axes[i], f"{name} (sil={res['metrics']['silhouette']:.3f})")
        plt.tight_layout()
        plt.show()
        return self

    def plot_silhouette_diagrams(self):
        # Classic silhouette diagrams for top models
        n_best = len(self.best_models)
        fig, axes = plt.subplots(1, n_best, figsize=(6*n_best, 5))
        if n_best == 1:
            axes = [axes]
        for i, (name, res) in enumerate(self.best_models):
            model = res['model']
            labels = model.predict(self.X)
            k = len(np.unique(labels))
            if k <= 1:
                axes[i].set_title(f"{name}: silhouette N/A")
                axes[i].axis('off')
                continue
            sil_values = silhouette_samples(self.X, labels)
            y_lower = 10
            for cluster in sorted(np.unique(labels)):
                ith_vals = sil_values[labels == cluster]
                ith_vals.sort()
                size = len(ith_vals)
                y_upper = y_lower + size
                color = plt.cm.tab10(cluster % 10)
                axes[i].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, facecolor=color, edgecolor=color, alpha=0.7)
                axes[i].text(-0.05, y_lower + 0.5 * size, str(cluster))
                y_lower = y_upper + 10
            axes[i].set_title(f"{name} silhouette (avg={res['metrics']['silhouette']:.3f})")
            axes[i].set_xlabel('Silhouette coefficient')
            axes[i].set_ylabel('Cluster label')
            axes[i].axvline(x=res['metrics']['silhouette'], color='red', linestyle='--')
            axes[i].set_yticks([])
            axes[i].set_xlim([-0.2, 1.0])
        plt.tight_layout()
        plt.show()
        return self

    def report(self):
        print("Clustering Report")
        for name, res in sorted(self.results.items(), key=lambda x: x[1]['metrics']['silhouette'], reverse=True):
            m = res['metrics']
            print(f"- {name}: silhouette={m['silhouette']:.3f}, DB={m['davies_bouldin']:.3f}, CH={m['calinski_harabasz']:.1f}")
        return self

    def run(self):
        (self.load_data()
         .initialize_models()
         .train_models()
         .select_best_models()
         .plot_best_clusters()
         .plot_silhouette_diagrams()
         .report())
        return True


def _infer_numeric_features(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['number']).columns.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run clustering analysis on a dataset")
    parser.add_argument("--dataset", type=str, default="dataset/iris_features.csv", help="Path to CSV with features only")
    parser.add_argument("--features", type=str, default="", help="Comma-separated feature columns; auto-infer if omitted")
    parser.add_argument("--labels", type=str, default="", help="Optional CSV path containing ground-truth labels (single column)")
    args = parser.parse_args()

    if args.features.strip():
        x_cols = [c.strip() for c in args.features.split(',') if c.strip()]
    else:
        df_preview = pd.read_csv(args.dataset) if os.path.exists(args.dataset) else pd.DataFrame()
        x_cols = _infer_numeric_features(df_preview) if not df_preview.empty else []

    analysis = ClusteringAnalysis(args.dataset, x_cols, args.labels or None)
    analysis.run()


if __name__ == "__main__":
    main()



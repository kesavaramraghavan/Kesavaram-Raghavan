#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from models.pca import PCAReducer
from models.svd import SVDReducer
from models.tsne import TSNEReducer


class DRAnalysis:
    def __init__(self, dataset_path: str, x_columns: List[str], n_components: int, labels_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.x_columns = x_columns
        self.n_components = n_components
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
            'PCA': PCAReducer(n_components=self.n_components),
            'SVD': SVDReducer(n_components=self.n_components),
            't-SNE': TSNEReducer(n_components=self.n_components),
        }
        return self

    def run_models(self):
        for name, reducer in self.models.items():
            embedding = reducer.fit_transform(self.X)
            metrics = reducer.evaluate(self.X, embedding)
            self.results[name] = {'embedding': embedding, 'metrics': metrics}
        return self

    def plot_embeddings(self):
        n = len(self.models)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
        if n == 1:
            axes = [axes]
        for i, (name, res) in enumerate(self.results.items()):
            emb = res['embedding']
            if self.n_components == 2:
                axes[i].scatter(emb[:, 0], emb[:, 1], c=self.labels if self.labels is not None else None, cmap='tab10', s=25)
                axes[i].set_title(f"{name} (trust={res['metrics']['trustworthiness']:.3f})")
            else:
                axes[i].scatter(emb[:, 0], emb[:, 1], s=25)
                axes[i].set_title(f"{name}")
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return self

    def report(self):
        print("Dimensionality Reduction Report")
        for name, res in sorted(self.results.items(), key=lambda x: x[1]['metrics']['trustworthiness'], reverse=True):
            m = res['metrics']
            print(f"- {name}: trustworthiness={m['trustworthiness']:.3f}, explained_variance={m['explained_variance']}")
        return self

    def run(self):
        (self.load_data()
         .initialize_models()
         .run_models()
         .plot_embeddings()
         .report())
        return True


def _infer_numeric_features(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['number']).columns.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run dimensionality reduction analysis on a dataset")
    parser.add_argument("--dataset", type=str, default="dataset/iris.csv", help="Path to CSV with features")
    parser.add_argument("--features", type=str, default="", help="Comma-separated feature columns; auto-infer if omitted")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components (2 or 3)")
    parser.add_argument("--labels", type=str, default="", help="Optional CSV path with labels for coloring")
    args = parser.parse_args()

    if args.features.strip():
        x_cols = [c.strip() for c in args.features.split(',') if c.strip()]
    else:
        df_preview = pd.read_csv(args.dataset) if os.path.exists(args.dataset) else pd.DataFrame()
        x_cols = _infer_numeric_features(df_preview) if not df_preview.empty else []

    analysis = DRAnalysis(args.dataset, x_cols, args.n_components, args.labels or None)
    analysis.run()


if __name__ == "__main__":
    main()


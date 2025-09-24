#!/usr/bin/env python3
# Clustering Learning Project

Unsupervised learning suite mirroring your regression/classification setup. Trains multiple clustering algorithms, scores them, selects top 3, and plots.

## Algorithms
- KMeans (`kmeans.py`)
- Agglomerative/Hierarchical (`agglomerative.py`)
- DBSCAN (`dbscan.py`)
- Gaussian Mixture (GMM) (`gmm.py`)

## Dataset
- Works with any CSV. If not provided, uses Iris features.

## Usage
```
python main.py --dataset dataset/iris.csv --features sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
```
If `--features` is omitted, numeric features are auto-inferred.

## Metrics
- Silhouette Score (higher is better)
- Davies-Bouldin Index (lower is better)
- Calinski-Harabasz Score (higher is better)
- Optional external metrics (if you provide `--labels`): ARI, NMI

## Plots
- 2D PCA scatter colored by cluster labels (for top 3 models)
- Silhouette comparison bars



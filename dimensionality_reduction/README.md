#!/usr/bin/env python3
# Dimensionality Reduction Learning Project

Reduce high-dimensional data into 2D/3D for visualization and learning. Ranks methods by trustworthiness and explained variance, and plots embeddings.

## Methods
- PCA (`pca.py`)
- TruncatedSVD (`svd.py`)
- t-SNE (`tsne.py`)

## Usage
```
python main.py --dataset dataset/iris.csv --features sepal length (cm),sepal width (cm),petal length (cm),petal width (cm) --n_components 2
```
If `--features` is omitted, numeric features are auto-inferred.

## Metrics
- Trustworthiness (higher is better)
- For linear methods (PCA/SVD): Explained variance ratio (sum)

## Plots
- 2D/3D scatter of embeddings (color by provided labels if `--labels` is passed)
- Bar comparison of trustworthiness and explained variance



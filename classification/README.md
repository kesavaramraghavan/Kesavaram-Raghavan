#!/usr/bin/env python3
# Classification Models Learning Project

This project mirrors the regression setup but for classification. It trains multiple classifiers, performs simple hyperparameter tuning and cross-validation, ranks models, selects the best 3, and plots comparisons.

## Models Included
- Logistic Regression (`logistic_regression.py`)
- Support Vector Machine (`svm_classifier.py`)
- Random Forest (`random_forest_classifier.py`)
- XGBoost (`xgboost_classifier.py`)
- K-Nearest Neighbors (`knn_classifier.py`)
- Naive Bayes (`naive_bayes_classifier.py`)
- Decision Tree (`decision_tree_classifier.py`)

## Dataset
- Default: Iris dataset (auto-downloaded from scikit-learn if `dataset/iris.csv` is missing)

## Structure
```
classification/
├── README.md
├── dataset/
├── models/
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── svm_classifier.py
│   ├── random_forest_classifier.py
│   ├── xgboost_classifier.py
│   ├── knn_classifier.py
│   ├── naive_bayes_classifier.py
│   └── decision_tree_classifier.py
└── main.py
```

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run: `python main.py`

Use with your own CSV:
```
python main.py --dataset path/to/data.csv --target target_col --features col1,col2,col3
```

If `--features` is omitted, numeric features are auto-inferred (excluding the target).

## Metrics and Plots
- Metrics: Accuracy, Precision (weighted), Recall (weighted), F1 (weighted), ROC-AUC (binary/multiclass OVR when supported)
- Plots: Overall comparison bars, training time, confusion matrices for top models, ROC curves when applicable



# Regression Models Learning Project

This project demonstrates various regression models using a publicly available dataset. It includes implementations of:

## Models Included:
1. **Linear Regression** (`linear_regression.py`)
2. **Polynomial Regression** (`polynomial_regression.py`)
3. **Ridge Regression** (`ridge_regression.py`)
4. **Lasso Regression** (`lasso_regression.py`)
5. **Elastic Net Regression** (`elastic_net_regression.py`)
6. **Support Vector Regression** (`svr_regression.py`)
7. **Random Forest Regression** (`random_forest_regression.py`)
8. **XGBoost Regression** (`xgboost_regression.py`)

## Dataset:
- **Boston Housing Dataset**: A classic dataset for regression problems
- Features: 13 numerical features (crime rate, zoning, etc.)
- Target: Median house values in Boston suburbs

## Project Structure:
```
regression/
├── README.md
├── dataset/
│   └── boston_housing.csv
├── models/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   ├── ridge_regression.py
│   ├── lasso_regression.py
│   ├── elastic_net_regression.py
│   ├── svr_regression.py
│   ├── random_forest_regression.py
│   └── xgboost_regression.py
├── main.py
└── requirements.txt
```

## Usage:
1. Install requirements: `pip install -r requirements.txt`
2. Run main.py: `python main.py`
3. The script will:
   - Load and preprocess the dataset
   - Train all regression models
   - Evaluate performance using multiple metrics
   - Select the top 3 best models
   - Create comprehensive visualizations

## Customization:
To use with your own dataset:
1. Replace the dataset file
2. Update the `X_COLUMNS` and `Y_COLUMN` variables in `main.py`
3. Run the script

## Features:
- **Gradient Descent**: Implemented for linear regression
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Cross-validation**: K-fold cross-validation for robust evaluation
- **Feature Scaling**: Standardization for better model performance
- **Performance Metrics**: R², MSE, MAE, RMSE
- **Visualizations**: Training curves, predictions vs actual, feature importance

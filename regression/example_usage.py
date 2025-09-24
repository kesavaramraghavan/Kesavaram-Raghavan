#!/usr/bin/env python3
"""
Example Usage Script
Demonstrates how to use individual regression models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import individual models
from models.linear_regression import LinearRegressionModel
from models.polynomial_regression import PolynomialRegressionModel
from models.ridge_regression import RidgeRegressionModel
from models.lasso_regression import LassoRegressionModel
from models.elastic_net_regression import ElasticNetRegressionModel
from models.svr_regression import SVRRegressionModel
from models.random_forest_regression import RandomForestRegressionModel
from models.xgboost_regression import XGBoostRegressionModel

def load_and_prepare_data():
    """
    Load and prepare the Boston Housing dataset
    """
    print("Loading Boston Housing dataset...")
    
    # Load data
    data = pd.read_csv("dataset/boston_housing.csv")
    
    # Define features and target
    X_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    y_column = 'MEDV'
    
    X = data[X_columns].values
    y = data[y_column].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_columns

def demonstrate_linear_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate Linear Regression with gradient descent
    """
    print("\n" + "="*60)
    print("LINEAR REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    lr_model = LinearRegressionModel(learning_rate=0.01, max_iterations=1000)
    
    # Train with gradient descent
    print("Training with Gradient Descent...")
    lr_model.fit_gradient_descent(X_train, y_train)
    
    # Train with sklearn (for comparison)
    print("Training with sklearn...")
    lr_model.fit_sklearn(X_train, y_train)
    
    # Evaluate
    test_metrics = lr_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    
    # Cross-validation
    cv_results = lr_model.cross_validate(X_train, y_train)
    print(f"\nCross-Validation:")
    print(f"Mean CV Score: {cv_results['mean_cv_score']:.4f}")
    print(f"Std CV Score: {cv_results['std_cv_score']:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    lr_model.plot_training_curve()
    lr_model.plot_predictions(X_test, y_test, "(Test Set)")
    lr_model.plot_feature_importance(feature_names)
    
    return lr_model

def demonstrate_polynomial_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate Polynomial Regression
    """
    print("\n" + "="*60)
    print("POLYNOMIAL REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    poly_model = PolynomialRegressionModel(max_degree=5)
    
    # Find best degree
    print("Finding best polynomial degree...")
    best_degree = poly_model.find_best_degree(X_train, y_train)
    
    # Train model
    print(f"Training with degree {best_degree}...")
    poly_model.fit(X_train, y_train)
    
    # Evaluate
    test_metrics = poly_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"Best Degree: {test_metrics['degree']}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    poly_model.plot_degree_selection()
    poly_model.plot_predictions(X_test, y_test, "(Test Set)")
    poly_model.plot_polynomial_curve(X_train, y_train, feature_idx=5)  # RM feature
    
    # Print equation
    poly_model.print_equation()
    
    return poly_model

def demonstrate_ridge_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate Ridge Regression
    """
    print("\n" + "="*60)
    print("RIDGE REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    ridge_model = RidgeRegressionModel()
    
    # Find best alpha
    print("Finding best alpha parameter...")
    best_alpha = ridge_model.find_best_alpha(X_train, y_train)
    
    # Train model
    print(f"Training with alpha {best_alpha}...")
    ridge_model.fit(X_train, y_train)
    
    # Evaluate
    test_metrics = ridge_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"Alpha: {test_metrics['alpha']}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    ridge_model.plot_alpha_selection()
    ridge_model.plot_predictions(X_test, y_test, "(Test Set)")
    ridge_model.plot_feature_importance(feature_names)
    ridge_model.plot_coefficient_path(X_train, y_train)
    
    # Compare with linear regression
    ridge_model.compare_with_linear(X_train, y_train)
    
    return ridge_model

def demonstrate_lasso_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate Lasso Regression
    """
    print("\n" + "="*60)
    print("LASSO REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    lasso_model = LassoRegressionModel()
    
    # Find best alpha
    print("Finding best alpha parameter...")
    best_alpha = lasso_model.find_best_alpha(X_train, y_train)
    
    # Train model
    print(f"Training with alpha {best_alpha}...")
    lasso_model.fit(X_train, y_train)
    
    # Evaluate
    test_metrics = lasso_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"Non-zero coefficients: {test_metrics['non_zero_coefs']}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    lasso_model.plot_alpha_selection()
    lasso_model.plot_predictions(X_test, y_test, "(Test Set)")
    lasso_model.plot_feature_importance(feature_names)
    lasso_model.plot_coefficient_path(X_train, y_train)
    
    # Analyze sparsity
    lasso_model.analyze_sparsity()
    
    return lasso_model

def demonstrate_elastic_net(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate Elastic Net Regression
    """
    print("\n" + "="*60)
    print("ELASTIC NET REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    elastic_model = ElasticNetRegressionModel()
    
    # Find best parameters
    print("Finding best parameters...")
    best_alpha, best_l1_ratio = elastic_model.find_best_parameters(X_train, y_train)
    
    # Train model
    print(f"Training with alpha {best_alpha}, l1_ratio {best_l1_ratio}...")
    elastic_model.fit(X_train, y_train)
    
    # Evaluate
    test_metrics = elastic_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"Alpha: {test_metrics['alpha']}")
    print(f"L1 Ratio: {test_metrics['l1_ratio']}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    elastic_model.plot_parameter_selection()
    elastic_model.plot_predictions(X_test, y_test, "(Test Set)")
    elastic_model.plot_feature_importance(feature_names)
    
    # Analyze sparsity
    elastic_model.analyze_sparsity()
    
    return elastic_model

def demonstrate_svr(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate Support Vector Regression
    """
    print("\n" + "="*60)
    print("SUPPORT VECTOR REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    svr_model = SVRRegressionModel(kernel='rbf')
    
    # Find best parameters
    print("Finding best parameters...")
    best_params = svr_model.find_best_parameters(X_train, y_train)
    
    # Train model
    print(f"Training with best parameters...")
    svr_model.fit(X_train, y_train)
    
    # Evaluate
    test_metrics = svr_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    svr_model.plot_predictions(X_test, y_test, "(Test Set)")
    svr_model.plot_support_vectors(X_train, y_train, feature_idx=5)
    
    # Analyze support vectors
    svr_model.analyze_support_vectors()
    
    # Compare kernels
    svr_model.compare_kernels(X_train, y_train)
    
    return svr_model

def demonstrate_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate Random Forest Regression
    """
    print("\n" + "="*60)
    print("RANDOM FOREST REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    rf_model = RandomForestRegressionModel()
    
    # Find best parameters
    print("Finding best parameters...")
    best_n_est, best_depth = rf_model.find_best_parameters(X_train, y_train)
    
    # Train model
    print(f"Training with best parameters...")
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    test_metrics = rf_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    rf_model.plot_parameter_selection()
    rf_model.plot_predictions(X_test, y_test, "(Test Set)")
    rf_model.plot_feature_importance(feature_names, top_n=8)
    rf_model.plot_trees_analysis()
    rf_model.plot_learning_curve(X_train, y_train)
    
    return rf_model

def demonstrate_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """
    Demonstrate XGBoost Regression
    """
    print("\n" + "="*60)
    print("XGBOOST REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Create model
    xgb_model = XGBoostRegressionModel()
    
    # Find best parameters
    print("Finding best parameters...")
    best_n_est, best_depth, best_lr = xgb_model.find_best_parameters(X_train, y_train)
    
    # Train model
    print(f"Training with best parameters...")
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    test_metrics = xgb_model.evaluate(X_test, y_test)
    print(f"\nTest Performance:")
    print(f"R² Score: {test_metrics['r2_score']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    xgb_model.plot_predictions(X_test, y_test, "(Test Set)")
    xgb_model.plot_feature_importance(feature_names, top_n=8)
    xgb_model.plot_learning_curve(X_train, y_train)
    xgb_model.plot_parameter_importance()
    
    return xgb_model

def main():
    """
    Main function to demonstrate all models
    """
    print("Regression Models Learning Project - Example Usage")
    print("=" * 60)
    print("This script demonstrates how to use individual regression models.")
    print("=" * 60)
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
        
        # Demonstrate each model
        models = {}
        
        # Linear Regression
        models['Linear'] = demonstrate_linear_regression(X_train, X_test, y_train, y_test, feature_names)
        
        # Polynomial Regression
        models['Polynomial'] = demonstrate_polynomial_regression(X_train, X_test, y_train, y_test, feature_names)
        
        # Ridge Regression
        models['Ridge'] = demonstrate_ridge_regression(X_train, X_test, y_train, y_test, feature_names)
        
        # Lasso Regression
        models['Lasso'] = demonstrate_lasso_regression(X_train, X_test, y_train, y_test, feature_names)
        
        # Elastic Net
        models['Elastic Net'] = demonstrate_elastic_net(X_train, X_test, y_train, y_test, feature_names)
        
        # SVR
        models['SVR'] = demonstrate_svr(X_train, X_test, y_train, y_test, feature_names)
        
        # Random Forest
        models['Random Forest'] = demonstrate_random_forest(X_train, X_test, y_train, y_test, feature_names)
        
        # XGBoost
        models['XGBoost'] = demonstrate_xgboost(X_train, X_test, y_train, y_test, feature_names)
        
        print("\n" + "="*60)
        print("ALL MODELS DEMONSTRATED SUCCESSFULLY!")
        print("="*60)
        
        # Summary of all models
        print("\nModel Summary:")
        print("-" * 40)
        for name, model in models.items():
            if hasattr(model, 'evaluate'):
                try:
                    metrics = model.evaluate(X_test, y_test)
                    print(f"{name:<20} R²: {metrics['r2_score']:.4f}")
                except:
                    print(f"{name:<20} Evaluation failed")
        
        print("\nExample usage completed!")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

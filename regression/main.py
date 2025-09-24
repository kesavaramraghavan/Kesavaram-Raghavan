#!/usr/bin/env python3
"""
Main Regression Analysis Script
Trains multiple regression models and finds the best performing ones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import argparse
from typing import List
import warnings
warnings.filterwarnings('ignore')

# Import all regression models
from models.linear_regression import LinearRegressionModel
from models.polynomial_regression import PolynomialRegressionModel
from models.ridge_regression import RidgeRegressionModel
from models.lasso_regression import LassoRegressionModel
from models.elastic_net_regression import ElasticNetRegressionModel
from models.svr_regression import SVRRegressionModel
from models.random_forest_regression import RandomForestRegressionModel
from models.xgboost_regression import XGBoostRegressionModel

class RegressionAnalysis:
    """
    Main class for regression analysis
    """
    
    def __init__(self, dataset_path, x_columns, y_column):
        self.dataset_path = dataset_path
        self.x_columns = x_columns
        self.y_column = y_column
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """
        Load and preprocess the dataset
        """
        print("Loading dataset...")
        self.data = pd.read_csv(self.dataset_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {self.x_columns}")
        print(f"Target: {self.y_column}")
        
        # Extract features and target
        self.X = self.data[self.x_columns].values
        self.y = self.data[self.y_column].values
        
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self
    
    def initialize_models(self):
        """
        Initialize all regression models
        """
        print("\nInitializing regression models...")
        
        self.models = {
            'Linear Regression': LinearRegressionModel(),
            'Polynomial Regression': PolynomialRegressionModel(max_degree=3),
            'Ridge Regression': RidgeRegressionModel(),
            'Lasso Regression': LassoRegressionModel(),
            'Elastic Net': ElasticNetRegressionModel(),
            'SVR (RBF)': SVRRegressionModel(kernel='rbf'),
            'Random Forest': RandomForestRegressionModel(),
            'XGBoost': XGBoostRegressionModel()
        }
        
        print(f"Initialized {len(self.models)} models")
        return self
    
    def train_models(self):
        """
        Train all models
        """
        print("\nTraining all models...")
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            try:
                # Train the model
                if name == 'Linear Regression':
                    # Train both gradient descent and sklearn versions
                    model.fit_gradient_descent(self.X_train, self.y_train)
                    model.fit_sklearn(self.X_train, self.y_train)
                else:
                    model.fit(self.X_train, self.y_train)
                
                # Evaluate on test set
                test_metrics = model.evaluate(self.X_test, self.y_test)
                
                # Cross-validation
                cv_results = model.cross_validate(self.X_train, self.y_train)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'test_metrics': test_metrics,
                    'cv_results': cv_results,
                    'training_time': time.time() - start_time
                }
                
                print(f"✓ {name} trained successfully!")
                print(f"  Test R²: {test_metrics['r2_score']:.4f}")
                print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
                print(f"  CV R²: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
                print(f"  Training time: {self.results[name]['training_time']:.2f}s")
                
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
                continue
        
        return self
    
    def select_best_models(self, n_best=3):
        """
        Select the best performing models
        """
        print(f"\n{'='*50}")
        print(f"Selecting top {n_best} models...")
        print(f"{'='*50}")
        
        # Sort models by test R² score
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_metrics']['r2_score'],
            reverse=True
        )
        
        print(f"\nModel Rankings (by Test R² Score):")
        print(f"{'Rank':<5} {'Model':<25} {'R² Score':<10} {'RMSE':<10} {'CV R²':<10}")
        print("-" * 70)
        
        for i, (name, result) in enumerate(sorted_models):
            rank = i + 1
            r2 = result['test_metrics']['r2_score']
            rmse = result['test_metrics']['rmse']
            cv_r2 = result['cv_results']['mean_cv_score']
            
            print(f"{rank:<5} {name:<25} {r2:<10.4f} {rmse:<10.4f} {cv_r2:<10.4f}")
        
        # Select top N models
        self.best_models = sorted_models[:n_best]
        
        print(f"\nTop {n_best} Models Selected:")
        for i, (name, result) in enumerate(self.best_models):
            print(f"{i+1}. {name} (R²: {result['test_metrics']['r2_score']:.4f})")
        
        return self
    
    def plot_model_comparison(self):
        """
        Create comprehensive model comparison plots
        """
        print("\nCreating model comparison plots...")
        
        # 1. Performance comparison
        self._plot_performance_comparison()
        
        # 2. Training time comparison
        self._plot_training_time_comparison()
        
        # 3. Best models detailed comparison
        self._plot_best_models_comparison()
        
        # 4. Predictions comparison for best models
        self._plot_predictions_comparison()
        
        return self
    
    def _plot_performance_comparison(self):
        """
        Plot overall performance comparison
        """
        models = list(self.results.keys())
        r2_scores = [self.results[name]['test_metrics']['r2_score'] for name in models]
        rmse_scores = [self.results[name]['test_metrics']['rmse'] for name in models]
        cv_scores = [self.results[name]['cv_results']['mean_cv_score'] for name in models]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # R² scores
        bars1 = ax1.bar(models, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('Test R² Scores')
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # RMSE scores
        bars2 = ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('RMSE')
        ax2.set_title('Test RMSE Scores')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # CV scores
        bars3 = ax3.bar(models, cv_scores, color='lightgreen', alpha=0.7)
        ax3.set_ylabel('CV R² Score')
        ax3.set_title('Cross-Validation R² Scores')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Highlight best models
        for i, (name, result) in enumerate(self.best_models):
            if name in models:
                idx = models.index(name)
                bars1[idx].set_color('gold')
                bars2[idx].set_color('gold')
                bars3[idx].set_color('gold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_training_time_comparison(self):
        """
        Plot training time comparison
        """
        models = list(self.results.keys())
        training_times = [self.results[name]['training_time'] for name in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, training_times, color='lightblue', alpha=0.7)
        plt.ylabel('Training Time (seconds)')
        plt.title('Model Training Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Highlight best models
        for i, (name, result) in enumerate(self.best_models):
            if name in models:
                idx = models.index(name)
                bars[idx].set_color('gold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_best_models_comparison(self):
        """
        Plot detailed comparison of best models
        """
        if not hasattr(self, 'best_models'):
            print("No best models selected yet.")
            return
        
        best_names = [name for name, _ in self.best_models]
        metrics = ['r2_score', 'rmse', 'mae']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [self.results[name]['test_metrics'][metric] for name in best_names]
            
            bars = axes[i].bar(best_names, values, color=['gold', 'silver', 'bronze'][:len(best_names)])
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'Best Models - {metric.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_predictions_comparison(self):
        """
        Plot predictions comparison for best models
        """
        if not hasattr(self, 'best_models'):
            print("No best models selected yet.")
            return
        
        n_best = len(self.best_models)
        fig, axes = plt.subplots(1, n_best, figsize=(6*n_best, 6))
        
        if n_best == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(self.best_models):
            model = result['model']
            y_pred = model.predict(self.X_test)
            
            axes[i].scatter(self.y_test, y_pred, alpha=0.6)
            axes[i].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'{name}\nR²: {result["test_metrics"]["r2_score"]:.4f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """
        Generate comprehensive analysis report
        """
        print(f"\n{'='*60}")
        print(f"REGRESSION ANALYSIS REPORT")
        print(f"{'='*60}")
        
        print(f"\nDataset Information:")
        print(f"  Path: {self.dataset_path}")
        print(f"  Features: {', '.join(self.x_columns)}")
        print(f"  Target: {self.y_column}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        
        print(f"\nModel Performance Summary:")
        print(f"{'Model':<25} {'Test R²':<10} {'Test RMSE':<12} {'CV R²':<10} {'Time (s)':<10}")
        print("-" * 80)
        
        for name, result in self.results.items():
            r2 = result['test_metrics']['r2_score']
            rmse = result['test_metrics']['rmse']
            cv_r2 = result['cv_results']['mean_cv_score']
            time_taken = result['training_time']
            
            print(f"{name:<25} {r2:<10.4f} {rmse:<12.4f} {cv_r2:<10.4f} {time_taken:<10.2f}")
        
        if hasattr(self, 'best_models'):
            print(f"\nTop {len(self.best_models)} Models:")
            for i, (name, result) in enumerate(self.best_models):
                print(f"  {i+1}. {name} (R²: {result['test_metrics']['r2_score']:.4f})")
        
        print(f"\nRecommendations:")
        if hasattr(self, 'best_models'):
            best_name, best_result = self.best_models[0]
            print(f"  • Best performing model: {best_name}")
            print(f"  • Consider using {best_name} for production with R² = {best_result['test_metrics']['r2_score']:.4f}")
            
            # Check for overfitting
            test_r2 = best_result['test_metrics']['r2_score']
            cv_r2 = best_result['cv_results']['mean_cv_score']
            if abs(test_r2 - cv_r2) > 0.1:
                print(f"  • Warning: Potential overfitting detected (test R²: {test_r2:.4f}, CV R²: {cv_r2:.4f})")
        
        print(f"\nAnalysis completed successfully!")
        return self
    
    def run_analysis(self):
        """
        Run the complete regression analysis
        """
        print("Starting Regression Analysis...")
        print("=" * 50)
        
        try:
            (self.load_data()
                 .initialize_models()
                 .train_models()
                 .select_best_models(3)
                 .plot_model_comparison()
                 .generate_report())
            
            print("\nAnalysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return False

def _infer_numeric_features(df: pd.DataFrame, target: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return [c for c in numeric_cols if c != target]

def main():
    """
    Main function to run the regression analysis
    """
    parser = argparse.ArgumentParser(description="Train multiple regression models and select the best ones")
    parser.add_argument("--dataset", type=str, default="dataset/boston_housing.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="MEDV", help="Target column name")
    parser.add_argument("--features", type=str, default="", help="Comma-separated feature columns. If omitted, auto-infer numeric features excluding target")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    Y_COLUMN = args.target
    if args.features.strip():
        X_COLUMNS = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        # Peek dataset once to infer numeric features
        _df_preview = pd.read_csv(DATASET_PATH, nrows=100)
        X_COLUMNS = _infer_numeric_features(_df_preview, Y_COLUMN)
    
    print("Regression Models Learning Project")
    print("=" * 50)
    print("This script will train multiple regression models and find the best performing ones.")
    print("Dataset:", DATASET_PATH)
    print("Target:", Y_COLUMN)
    print("Features:", ', '.join(X_COLUMNS))
    print("=" * 50)
    
    # Create and run analysis
    analysis = RegressionAnalysis(DATASET_PATH, X_COLUMNS, Y_COLUMN)
    success = analysis.run_analysis()
    
    if success:
        print("\nTo use with your own dataset:")
        print("1. Update DATASET_PATH, X_COLUMNS, and Y_COLUMN in main()")
        print("2. Ensure your dataset has the specified columns")
        print("3. Run the script again")
        
        # Ask user if they want to see specific model details
        print("\nWould you like to explore specific models?")
        print("Available models:", ', '.join(analysis.models.keys()))
        
        # Example: Show feature importance for best model
        if hasattr(analysis, 'best_models') and analysis.best_models:
            best_name, best_result = analysis.best_models[0]
            best_model = best_result['model']
            
            print(f"\nShowing feature importance for best model: {best_name}")
            try:
                if hasattr(best_model, 'plot_feature_importance'):
                    best_model.plot_feature_importance(feature_names=X_COLUMNS)
                if hasattr(best_model, 'plot_training_curve'):
                    best_model.plot_training_curve()
            except Exception as e:
                print(f"Could not show additional plots: {e}")

if __name__ == "__main__":
    main()

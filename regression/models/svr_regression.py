import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

class SVRRegressionModel:
    """
    Support Vector Regression Model with hyperparameter tuning
    """
    
    def __init__(self, kernel='rbf'):
        self.kernel = kernel
        self.best_model = None
        self.model_name = f"SVR ({kernel.upper()})"
        self.best_params = {}
        
    def find_best_parameters(self, X, y, cv=5):
        """
        Find the best hyperparameters using cross-validation
        """
        print(f"Finding best parameters for {self.model_name}...")
        
        if self.kernel == 'rbf':
            # For RBF kernel, tune C and gamma
            C_range = [0.1, 1, 10, 100]
            gamma_range = ['scale', 'auto', 0.001, 0.01, 0.1]
        elif self.kernel == 'linear':
            # For linear kernel, only tune C
            C_range = [0.1, 1, 10, 100]
            gamma_range = ['scale']
        else:
            # For polynomial kernel, tune C, gamma, and degree
            C_range = [0.1, 1, 10, 100]
            gamma_range = ['scale', 'auto', 0.001, 0.01, 0.1]
        
        best_score = -np.inf
        
        for C in C_range:
            for gamma in gamma_range:
                # Create and evaluate model
                model = SVR(kernel=self.kernel, C=C, gamma=gamma)
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                
                mean_score = scores.mean()
                
                print(f"C={C}, gamma={gamma}: CV Score = {mean_score:.4f} ± {scores.std():.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_params = {'C': C, 'gamma': gamma}
        
        print(f"\nBest parameters: C = {self.best_params['C']}, gamma = {self.best_params['gamma']}")
        
        return self.best_params
    
    def fit(self, X, y, params=None):
        """
        Fit the SVR model
        """
        if params is None:
            if not self.best_params:
                self.find_best_parameters(X, y)
            params = self.best_params
        
        # Create and fit the model
        self.best_model = SVR(
            kernel=self.kernel, 
            C=params['C'], 
            gamma=params['gamma']
        )
        self.best_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.best_model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model
        """
        y_pred = self.predict(X)
        
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'C': self.best_params.get('C'),
            'gamma': self.best_params.get('gamma')
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        if self.best_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        scores = cross_val_score(self.best_model, X, y, cv=cv, scoring='r2')
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'cv_scores': scores
        }
    
    def plot_predictions(self, X, y, title_suffix=""):
        """
        Plot actual vs predicted values
        """
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.model_name} - Actual vs Predicted {title_suffix}')
        plt.grid(True)
        plt.show()
    
    def plot_support_vectors(self, X, y, feature_idx=0):
        """
        Plot support vectors for a single feature
        """
        if X.shape[1] == 1:
            X_plot = X
        else:
            X_plot = X[:, feature_idx:feature_idx+1]
        
        # Get support vectors
        support_vectors = self.best_model.support_vectors_
        if support_vectors.shape[1] == 1:
            sv_plot = support_vectors
        else:
            sv_plot = support_vectors[:, feature_idx:feature_idx+1]
        
        # Sort for plotting
        sort_idx = np.argsort(X_plot.flatten())
        X_sorted = X_plot[sort_idx]
        y_sorted = y[sort_idx]
        
        # Predictions
        y_pred = self.predict(X_sorted)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(X_sorted, y_sorted, alpha=0.6, label='Data Points')
        plt.scatter(sv_plot, self.best_model.predict(sv_plot), 
                   color='red', s=100, label='Support Vectors', zorder=5)
        plt.plot(X_sorted, y_pred, 'g-', linewidth=2, label='SVR Prediction')
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Target')
        plt.title(f'{self.model_name} - Support Vectors and Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_support_vector_info(self):
        """
        Get information about support vectors
        """
        if self.best_model is None:
            return None
        
        n_support_vectors = len(self.best_model.support_vectors_)
        n_samples = self.best_model.n_samples_in_
        
        return {
            'n_support_vectors': n_support_vectors,
            'n_samples': n_samples,
            'support_vector_ratio': n_support_vectors / n_samples,
            'support_vector_indices': self.best_model.support_
        }
    
    def analyze_support_vectors(self):
        """
        Analyze the support vectors
        """
        info = self.get_support_vector_info()
        if info is None:
            print("Model not fitted yet. Call fit() first.")
            return
        
        print(f"\nSupport Vector Analysis for {self.model_name}:")
        print(f"Total samples: {info['n_samples']}")
        print(f"Support vectors: {info['n_support_vectors']}")
        print(f"Support vector ratio: {info['support_vector_ratio']:.2%}")
        
        # Plot support vector ratio
        plt.figure(figsize=(10, 6))
        labels = ['Support Vectors', 'Other Samples']
        sizes = [info['n_support_vectors'], info['n_samples'] - info['n_support_vectors']]
        colors = ['lightcoral', 'lightblue']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'{self.model_name} - Support Vector Distribution')
        plt.axis('equal')
        plt.show()
    
    def compare_kernels(self, X, y):
        """
        Compare different kernel performances
        """
        kernels = ['linear', 'rbf', 'poly']
        results = {}
        
        for kernel in kernels:
            # Use default parameters for fair comparison
            model = SVR(kernel=kernel)
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            results[kernel] = {
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
        
        # Print comparison
        print(f"\nKernel Comparison:")
        print(f"{'Kernel':<10} {'R² Score':<10} {'Std':<10}")
        print("-" * 30)
        for kernel, metrics in results.items():
            print(f"{kernel:<10} {metrics['mean_score']:<10.4f} {metrics['std_score']:<10.4f}")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        kernels = list(results.keys())
        mean_scores = [results[k]['mean_score'] for k in kernels]
        std_scores = [results[k]['std_score'] for k in kernels]
        
        plt.bar(kernels, mean_scores, yerr=std_scores, capsize=5, 
               color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.ylabel('Cross-Validation R² Score')
        plt.title('SVR Kernel Performance Comparison')
        plt.grid(True, alpha=0.3)
        plt.show()

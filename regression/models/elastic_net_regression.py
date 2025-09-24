import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

class ElasticNetRegressionModel:
    """
    Elastic Net Regression Model combining L1 and L2 regularization
    """
    
    def __init__(self, alpha_range=None, l1_ratio_range=None):
        if alpha_range is None:
            self.alpha_range = [0.001, 0.01, 0.1, 1, 10, 100]
        else:
            self.alpha_range = alpha_range
            
        if l1_ratio_range is None:
            self.l1_ratio_range = [0.1, 0.3, 0.5, 0.7, 0.9]
        else:
            self.l1_ratio_range = l1_ratio_range
            
        self.best_alpha = None
        self.best_l1_ratio = None
        self.best_model = None
        self.model_name = "Elastic Net Regression"
        self.parameter_scores = {}
        
    def find_best_parameters(self, X, y, cv=5):
        """
        Find the best alpha and l1_ratio parameters using cross-validation
        """
        print(f"Finding best parameters for {self.model_name}...")
        
        best_score = -np.inf
        
        for alpha in self.alpha_range:
            for l1_ratio in self.l1_ratio_range:
                # Create and evaluate model
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                
                mean_score = scores.mean()
                
                self.parameter_scores[(alpha, l1_ratio)] = {
                    'mean_score': mean_score,
                    'std_score': scores.std(),
                    'scores': scores
                }
                
                print(f"Alpha {alpha}, L1_ratio {l1_ratio}: CV Score = {mean_score:.4f} ± {scores.std():.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_alpha = alpha
                    self.best_l1_ratio = l1_ratio
        
        print(f"\nBest parameters: Alpha = {self.best_alpha}, L1_ratio = {self.best_l1_ratio}")
        
        return self.best_alpha, self.best_l1_ratio
    
    def fit(self, X, y, alpha=None, l1_ratio=None):
        """
        Fit the elastic net regression model
        """
        if alpha is None or l1_ratio is None:
            if self.best_alpha is None:
                self.find_best_parameters(X, y)
            alpha = self.best_alpha
            l1_ratio = self.best_l1_ratio
        
        # Create and fit the model
        self.best_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
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
            'alpha': self.best_alpha,
            'l1_ratio': self.best_l1_ratio,
            'non_zero_coefs': np.sum(self.best_model.coef_ != 0)
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
    
    def plot_parameter_selection(self):
        """
        Plot the parameter selection results
        """
        if not self.parameter_scores:
            print("No parameter scores available. Run find_best_parameters() first.")
            return
        
        # Create heatmap
        alphas = sorted(list(set([params[0] for params in self.parameter_scores.keys()])))
        l1_ratios = sorted(list(set([params[1] for params in self.parameter_scores.keys()])))
        
        scores_matrix = np.zeros((len(l1_ratios), len(alphas)))
        
        for i, l1_ratio in enumerate(l1_ratios):
            for j, alpha in enumerate(alphas):
                if (alpha, l1_ratio) in self.parameter_scores:
                    scores_matrix[i, j] = self.parameter_scores[(alpha, l1_ratio)]['mean_score']
        
        plt.figure(figsize=(12, 8))
        im = plt.imshow(scores_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Cross-Validation R² Score')
        
        # Set labels
        plt.xticks(range(len(alphas)), [f'{a:.3f}' for a in alphas])
        plt.yticks(range(len(l1_ratios)), [f'{r:.1f}' for r in l1_ratios])
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('L1 Ratio (Lasso vs Ridge)')
        plt.title(f'{self.model_name} - Parameter Selection Heatmap')
        
        # Highlight best parameters
        best_alpha_idx = alphas.index(self.best_alpha)
        best_l1_ratio_idx = l1_ratios.index(self.best_l1_ratio)
        plt.plot(best_alpha_idx, best_l1_ratio_idx, 'r*', markersize=15, label=f'Best: α={self.best_alpha}, L1={self.best_l1_ratio}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
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
        plt.title(f'{self.model_name} (α={self.best_alpha}, L1={self.best_l1_ratio}) - Actual vs Predicted {title_suffix}')
        plt.grid(True)
        plt.show()
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (coefficients)
        """
        if self.best_model is None:
            return None
            
        coefs = self.best_model.coef_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(coefs))]
            
        importance = dict(zip(feature_names, coefs))
        return importance
    
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance
        """
        importance = self.get_feature_importance(feature_names)
        if importance is None:
            return
        
        plt.figure(figsize=(12, 6))
        features = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(features, values)
        plt.xlabel('Coefficient Value')
        plt.title(f'{self.model_name} (α={self.best_alpha}, L1={self.best_l1_ratio}) - Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_sparsity(self):
        """
        Analyze the sparsity of the model
        """
        if self.best_model is None:
            print("Model not fitted yet. Call fit() first.")
            return
        
        coefs = self.best_model.coef_
        total_features = len(coefs)
        non_zero_features = np.sum(coefs != 0)
        zero_features = total_features - non_zero_features
        
        print(f"\nSparsity Analysis for {self.model_name}:")
        print(f"Total features: {total_features}")
        print(f"Non-zero coefficients: {non_zero_features}")
        print(f"Zero coefficients: {zero_features}")
        print(f"Sparsity: {zero_features/total_features*100:.2f}%")
        print(f"Alpha: {self.best_alpha}")
        print(f"L1 Ratio: {self.best_l1_ratio}")
        
        # Plot sparsity
        plt.figure(figsize=(10, 6))
        labels = ['Non-zero', 'Zero']
        sizes = [non_zero_features, zero_features]
        colors = ['lightblue', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'{self.model_name} - Coefficient Sparsity')
        plt.axis('equal')
        plt.show()
    
    def compare_with_other_models(self, X, y):
        """
        Compare Elastic Net with Linear, Ridge, and Lasso
        """
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        
        # Fit all models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=1.0, random_state=42, max_iter=2000),
            'Elastic Net': self.best_model
        }
        
        results = {}
        for name, model in models.items():
            if name == 'Elastic Net':
                y_pred = self.predict(X)
            else:
                model.fit(X, y)
                y_pred = model.predict(X)
            
            results[name] = {
                'r2': r2_score(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred))
            }
        
        # Print comparison
        print(f"\nModel Comparison:")
        print(f"{'Model':<15} {'R² Score':<10} {'RMSE':<10}")
        print("-" * 35)
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['r2']:<10.4f} {metrics['rmse']:<10.4f}")
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² comparison
        names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in names]
        ax1.bar(names, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.grid(True, alpha=0.3)
        
        # RMSE comparison
        rmse_scores = [results[name]['rmse'] for name in names]
        ax2.bar(names, rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

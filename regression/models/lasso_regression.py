import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

class LassoRegressionModel:
    """
    Lasso Regression Model with L1 regularization
    """
    
    def __init__(self, alpha_range=None):
        if alpha_range is None:
            self.alpha_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        else:
            self.alpha_range = alpha_range
            
        self.best_alpha = None
        self.best_model = None
        self.model_name = "Lasso Regression"
        self.alpha_scores = {}
        
    def find_best_alpha(self, X, y, cv=5):
        """
        Find the best alpha parameter using cross-validation
        """
        print(f"Finding best alpha parameter for {self.model_name}...")
        
        for alpha in self.alpha_range:
            # Create and evaluate model
            model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            self.alpha_scores[alpha] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"Alpha {alpha}: CV Score = {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Find best alpha
        best_alpha = max(self.alpha_scores.keys(), 
                        key=lambda a: self.alpha_scores[a]['mean_score'])
        
        self.best_alpha = best_alpha
        print(f"\nBest alpha: {best_alpha}")
        
        return best_alpha
    
    def fit(self, X, y, alpha=None):
        """
        Fit the lasso regression model
        """
        if alpha is None:
            if self.best_alpha is None:
                self.find_best_alpha(X, y)
            alpha = self.best_alpha
        
        # Create and fit the model
        self.best_model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
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
    
    def plot_alpha_selection(self):
        """
        Plot the alpha selection results
        """
        if not self.alpha_scores:
            print("No alpha scores available. Run find_best_alpha() first.")
            return
        
        alphas = list(self.alpha_scores.keys())
        mean_scores = [self.alpha_scores[a]['mean_score'] for a in alphas]
        std_scores = [self.alpha_scores[a]['std_score'] for a in alphas]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(alphas, mean_scores, yerr=std_scores, marker='o', capsize=5)
        plt.axvline(x=self.best_alpha, color='red', linestyle='--', 
                   label=f'Best Alpha: {self.best_alpha}')
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('Cross-Validation R² Score')
        plt.title(f'{self.model_name} - Alpha Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
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
        plt.title(f'{self.model_name} (α={self.best_alpha}) - Actual vs Predicted {title_suffix}')
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
        plt.title(f'{self.model_name} (α={self.best_alpha}) - Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_coefficient_path(self, X, y, alpha_range=None):
        """
        Plot coefficient paths for different alpha values
        """
        if alpha_range is None:
            alpha_range = np.logspace(-3, 3, 20)
        
        coefs = []
        for alpha in alpha_range:
            model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
            model.fit(X, y)
            coefs.append(model.coef_)
        
        coefs = np.array(coefs)
        
        plt.figure(figsize=(12, 8))
        for i in range(coefs.shape[1]):
            plt.plot(alpha_range, coefs[:, i], label=f'Feature {i}')
        
        plt.axvline(x=self.best_alpha, color='red', linestyle='--', 
                   label=f'Best Alpha: {self.best_alpha}')
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('Coefficient Values')
        plt.title(f'{self.model_name} - Coefficient Paths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
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
        
        # Plot sparsity
        plt.figure(figsize=(10, 6))
        labels = ['Non-zero', 'Zero']
        sizes = [non_zero_features, zero_features]
        colors = ['lightblue', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'{self.model_name} - Coefficient Sparsity')
        plt.axis('equal')
        plt.show()
    
    def compare_with_linear(self, X, y):
        """
        Compare Lasso with Linear Regression
        """
        from sklearn.linear_model import LinearRegression
        
        # Fit linear regression
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_pred = linear_model.predict(X)
        linear_r2 = r2_score(y, linear_pred)
        
        # Lasso predictions
        lasso_pred = self.predict(X)
        lasso_r2 = r2_score(y, lasso_pred)
        
        # Compare coefficients
        linear_coefs = linear_model.coef_
        lasso_coefs = self.best_model.coef_
        
        print(f"\nComparison with Linear Regression:")
        print(f"Linear R²: {linear_r2:.4f}")
        print(f"Lasso R²: {lasso_r2:.4f}")
        print(f"Improvement: {lasso_r2 - linear_r2:.4f}")
        
        # Plot coefficient comparison
        plt.figure(figsize=(12, 6))
        x = np.arange(len(linear_coefs))
        width = 0.35
        
        plt.bar(x - width/2, linear_coefs, width, label='Linear Regression', alpha=0.7)
        plt.bar(x + width/2, lasso_coefs, width, label=f'Lasso (α={self.best_alpha})', alpha=0.7)
        
        plt.xlabel('Features')
        plt.ylabel('Coefficient Values')
        plt.title('Coefficient Comparison: Linear vs Lasso Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

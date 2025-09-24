import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import time

class PolynomialRegressionModel:
    """
    Polynomial Regression Model with automatic degree selection
    """
    
    def __init__(self, max_degree=5):
        self.max_degree = max_degree
        self.best_degree = None
        self.best_model = None
        self.model_name = "Polynomial Regression"
        self.degree_scores = {}
        
    def find_best_degree(self, X, y, cv=5):
        """
        Find the best polynomial degree using cross-validation
        """
        print(f"Finding best polynomial degree (1 to {self.max_degree})...")
        
        for degree in range(1, self.max_degree + 1):
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            
            # Create and evaluate model
            model = LinearRegression()
            scores = cross_val_score(model, X_poly, y, cv=cv, scoring='r2')
            
            self.degree_scores[degree] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"Degree {degree}: CV Score = {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Find best degree
        best_degree = max(self.degree_scores.keys(), 
                         key=lambda d: self.degree_scores[d]['mean_score'])
        
        self.best_degree = best_degree
        print(f"\nBest degree: {best_degree}")
        
        return best_degree
    
    def fit(self, X, y, degree=None):
        """
        Fit the polynomial regression model
        """
        if degree is None:
            if self.best_degree is None:
                self.find_best_degree(X, y)
            degree = self.best_degree
        
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        
        # Create pipeline
        self.best_model = Pipeline([
            ('poly', poly_features),
            ('linear', LinearRegression())
        ])
        
        # Fit the model
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
            'degree': self.best_degree
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
    
    def plot_degree_selection(self):
        """
        Plot the degree selection results
        """
        if not self.degree_scores:
            print("No degree scores available. Run find_best_degree() first.")
            return
        
        degrees = list(self.degree_scores.keys())
        mean_scores = [self.degree_scores[d]['mean_score'] for d in degrees]
        std_scores = [self.degree_scores[d]['std_score'] for d in degrees]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(degrees, mean_scores, yerr=std_scores, marker='o', capsize=5)
        plt.axvline(x=self.best_degree, color='red', linestyle='--', 
                   label=f'Best Degree: {self.best_degree}')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Cross-Validation R² Score')
        plt.title(f'{self.model_name} - Degree Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
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
        plt.title(f'{self.model_name} (Degree {self.best_degree}) - Actual vs Predicted {title_suffix}')
        plt.grid(True)
        plt.show()
    
    def plot_polynomial_curve(self, X, y, feature_idx=0):
        """
        Plot polynomial curve for a single feature
        """
        if X.shape[1] == 1:
            # Single feature case
            X_plot = X
        else:
            # Multiple features - use the specified feature
            X_plot = X[:, feature_idx:feature_idx+1]
        
        # Sort for plotting
        sort_idx = np.argsort(X_plot.flatten())
        X_sorted = X_plot[sort_idx]
        y_sorted = y[sort_idx]
        
        # Predictions
        y_pred = self.predict(X_sorted)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(X_sorted, y_sorted, alpha=0.6, label='Actual Data')
        plt.plot(X_sorted, y_pred, 'r-', linewidth=2, label=f'Polynomial (Degree {self.best_degree})')
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Target')
        plt.title(f'{self.model_name} - Polynomial Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_coefficients(self):
        """
        Get polynomial coefficients
        """
        if self.best_model is None:
            return None
        
        # Get feature names
        poly_features = self.best_model.named_steps['poly']
        feature_names = poly_features.get_feature_names_out()
        
        # Get coefficients
        coefficients = self.best_model.named_steps['linear'].coef_
        intercept = self.best_model.named_steps['linear'].intercept_
        
        return {
            'feature_names': feature_names,
            'coefficients': coefficients,
            'intercept': intercept
        }
    
    def print_equation(self):
        """
        Print the polynomial equation
        """
        coef_info = self.get_coefficients()
        if coef_info is None:
            return
        
        print(f"\n{self.model_name} Equation (Degree {self.best_degree}):")
        print(f"y = {coef_info['intercept']:.4f}")
        
        for i, (name, coef) in enumerate(zip(coef_info['feature_names'], coef_info['coefficients'])):
            if abs(coef) > 1e-10:  # Only show non-zero coefficients
                if coef >= 0:
                    print(f"     + {coef:.4f} * {name}")
                else:
                    print(f"     - {abs(coef):.4f} * {name}")

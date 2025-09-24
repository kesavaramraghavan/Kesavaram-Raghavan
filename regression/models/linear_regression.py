import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

class LinearRegressionModel:
    """
    Linear Regression Model with Gradient Descent implementation
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.sklearn_model = LinearRegression()
        self.model_name = "Linear Regression"
        
    def fit_gradient_descent(self, X, y):
        """
        Fit the model using gradient descent
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Check convergence
            if len(self.cost_history) > 1:
                if abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                    break
        
        return self
    
    def fit_sklearn(self, X, y):
        """
        Fit using sklearn's implementation
        """
        self.sklearn_model.fit(X, y)
        return self
    
    def predict(self, X, use_sklearn=True):
        """
        Make predictions
        """
        if use_sklearn and hasattr(self.sklearn_model, "coef_"):
            return self.sklearn_model.predict(X)
        else:
            return np.dot(X, self.weights) + self.bias
    
    def evaluate(self, X, y):
        """
        Evaluate the model
        """
        y_pred = self.predict(X)
        
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        scores = cross_val_score(self.sklearn_model, X, y, cv=cv, scoring='r2')
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'cv_scores': scores
        }
    
    def plot_training_curve(self):
        """
        Plot the training curve (cost vs iterations)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title(f'{self.model_name} - Training Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
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
        plt.title(f'{self.model_name} - Actual vs Predicted {title_suffix}')
        plt.grid(True)
        plt.show()
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (coefficients)
        """
        if self.sklearn_model.coef_ is not None:
            coefs = self.sklearn_model.coef_
        else:
            coefs = self.weights
            
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(coefs))]
            
        importance = dict(zip(feature_names, coefs))
        return importance
    
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance
        """
        importance = self.get_feature_importance(feature_names)
        
        plt.figure(figsize=(12, 6))
        features = list(importance.keys())
        values = list(importance.values())
        
        plt.barh(features, values)
        plt.xlabel('Coefficient Value')
        plt.title(f'{self.model_name} - Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

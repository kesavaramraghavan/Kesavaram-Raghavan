import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

class RandomForestRegressionModel:
    """
    Random Forest Regression Model with hyperparameter tuning
    """
    
    def __init__(self, n_estimators_range=None, max_depth_range=None):
        if n_estimators_range is None:
            self.n_estimators_range = [50, 100, 200, 300]
        else:
            self.n_estimators_range = n_estimators_range
            
        if max_depth_range is None:
            self.max_depth_range = [None, 5, 10, 15, 20]
        else:
            self.max_depth_range = max_depth_range
            
        self.best_n_estimators = None
        self.best_max_depth = None
        self.best_model = None
        self.model_name = "Random Forest Regression"
        self.parameter_scores = {}
        
    def find_best_parameters(self, X, y, cv=5):
        """
        Find the best hyperparameters using cross-validation
        """
        print(f"Finding best parameters for {self.model_name}...")
        
        best_score = -np.inf
        
        for n_estimators in self.n_estimators_range:
            for max_depth in self.max_depth_range:
                # Create and evaluate model
                model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=42
                )
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                
                mean_score = scores.mean()
                
                self.parameter_scores[(n_estimators, max_depth)] = {
                    'mean_score': mean_score,
                    'std_score': scores.std(),
                    'scores': scores
                }
                
                depth_str = str(max_depth) if max_depth is not None else 'None'
                print(f"n_estimators={n_estimators}, max_depth={depth_str}: CV Score = {mean_score:.4f} ± {scores.std():.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_n_estimators = n_estimators
                    self.best_max_depth = max_depth
        
        depth_str = str(self.best_max_depth) if self.best_max_depth is not None else 'None'
        print(f"\nBest parameters: n_estimators = {self.best_n_estimators}, max_depth = {depth_str}")
        
        return self.best_n_estimators, self.best_max_depth
    
    def fit(self, X, y, n_estimators=None, max_depth=None):
        """
        Fit the Random Forest model
        """
        if n_estimators is None or max_depth is None:
            if self.best_n_estimators is None:
                self.find_best_parameters(X, y)
            n_estimators = self.best_n_estimators
            max_depth = self.best_max_depth
        
        # Create and fit the model
        self.best_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
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
            'n_estimators': self.best_n_estimators,
            'max_depth': self.best_max_depth
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
        n_estimators_list = sorted(list(set([params[0] for params in self.parameter_scores.keys()])))
        max_depths_list = sorted(list(set([params[1] for params in self.parameter_scores.keys()])))
        
        scores_matrix = np.zeros((len(max_depths_list), len(n_estimators_list)))
        
        for i, max_depth in enumerate(max_depths_list):
            for j, n_estimators in enumerate(n_estimators_list):
                if (n_estimators, max_depth) in self.parameter_scores:
                    scores_matrix[i, j] = self.parameter_scores[(n_estimators, max_depth)]['mean_score']
        
        plt.figure(figsize=(12, 8))
        im = plt.imshow(scores_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Cross-Validation R² Score')
        
        # Set labels
        plt.xticks(range(len(n_estimators_list)), n_estimators_list)
        plt.yticks(range(len(max_depths_list)), [str(d) if d is not None else 'None' for d in max_depths_list])
        plt.xlabel('Number of Estimators')
        plt.ylabel('Max Depth')
        plt.title(f'{self.model_name} - Parameter Selection Heatmap')
        
        # Highlight best parameters
        best_n_est_idx = n_estimators_list.index(self.best_n_estimators)
        best_depth_idx = max_depths_list.index(self.best_max_depth)
        plt.plot(best_n_est_idx, best_depth_idx, 'r*', markersize=15, 
                label=f'Best: n_est={self.best_n_estimators}, depth={self.best_max_depth}')
        
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
        plt.title(f'{self.model_name} - Actual vs Predicted {title_suffix}')
        plt.grid(True)
        plt.show()
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance
        """
        if self.best_model is None:
            return None
            
        importance = self.best_model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
        importance_dict = dict(zip(feature_names, importance))
        return importance_dict
    
    def plot_feature_importance(self, feature_names=None, top_n=10):
        """
        Plot feature importance
        """
        importance = self.get_feature_importance(feature_names)
        if importance is None:
            return
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = sorted_importance[:top_n]
        
        plt.figure(figsize=(12, 8))
        features = [item[0] for item in top_features]
        values = [item[1] for item in top_features]
        
        plt.barh(features, values)
        plt.xlabel('Feature Importance')
        plt.title(f'{self.model_name} - Top {top_n} Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_trees_analysis(self):
        """
        Analyze individual trees in the forest
        """
        if self.best_model is None:
            print("Model not fitted yet. Call fit() first.")
            return
        
        n_trees = self.best_n_estimators
        tree_depths = [tree.tree_.max_depth for tree in self.best_model.estimators_]
        tree_leaves = [tree.tree_.n_leaves for tree in self.best_model.estimators_]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Tree depths distribution
        ax1.hist(tree_depths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Tree Depth')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Tree Depths')
        ax1.grid(True, alpha=0.3)
        
        # Tree leaves distribution
        ax2.hist(tree_leaves, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Number of Leaves')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Tree Leaves')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nTree Analysis for {self.model_name}:")
        print(f"Average tree depth: {np.mean(tree_depths):.2f}")
        print(f"Average tree leaves: {np.mean(tree_leaves):.2f}")
        print(f"Total number of trees: {n_trees}")
    
    def plot_learning_curve(self, X, y):
        """
        Plot learning curve showing performance vs number of trees
        """
        if self.best_model is None:
            print("Model not fitted yet. Call fit() first.")
            return
        
        n_estimators_range = [10, 25, 50, 75, 100, 150, 200, 250, 300]
        train_scores = []
        val_scores = []
        
        for n_est in n_estimators_range:
            # Create model with current number of estimators
            model = RandomForestRegressor(
                n_estimators=n_est, 
                max_depth=self.best_max_depth, 
                random_state=42
            )
            
            # Fit and evaluate
            model.fit(X, y)
            train_pred = model.predict(X)
            train_score = r2_score(y, train_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            val_score = cv_scores.mean()
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        plt.figure(figsize=(12, 6))
        plt.plot(n_estimators_range, train_scores, 'o-', label='Training Score', color='blue')
        plt.plot(n_estimators_range, val_scores, 'o-', label='Cross-Validation Score', color='red')
        plt.axvline(x=self.best_n_estimators, color='green', linestyle='--', 
                   label=f'Best n_estimators: {self.best_n_estimators}')
        plt.xlabel('Number of Estimators')
        plt.ylabel('R² Score')
        plt.title(f'{self.model_name} - Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_with_linear(self, X, y):
        """
        Compare Random Forest with Linear Regression
        """
        from sklearn.linear_model import LinearRegression
        
        # Fit linear regression
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_pred = linear_model.predict(X)
        linear_r2 = r2_score(y, linear_pred)
        
        # Random Forest predictions
        rf_pred = self.predict(X)
        rf_r2 = r2_score(y, rf_pred)
        
        print(f"\nComparison with Linear Regression:")
        print(f"Linear R²: {linear_r2:.4f}")
        print(f"Random Forest R²: {rf_r2:.4f}")
        print(f"Improvement: {rf_r2 - linear_r2:.4f}")
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² comparison
        models = ['Linear', 'Random Forest']
        r2_scores = [linear_r2, rf_r2]
        colors = ['skyblue', 'lightgreen']
        
        ax1.bar(models, r2_scores, color=colors)
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Predictions comparison
        ax2.scatter(y, linear_pred, alpha=0.6, label='Linear Regression')
        ax2.scatter(y, rf_pred, alpha=0.6, label='Random Forest')
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('Predictions Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

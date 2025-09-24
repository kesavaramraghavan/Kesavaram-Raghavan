import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import time

class XGBoostRegressionModel:
    """
    XGBoost Regression Model with hyperparameter tuning
    """
    
    def __init__(self, n_estimators_range=None, max_depth_range=None, learning_rate_range=None):
        if n_estimators_range is None:
            self.n_estimators_range = [50, 100, 200, 300]
        else:
            self.n_estimators_range = n_estimators_range
            
        if max_depth_range is None:
            self.max_depth_range = [3, 5, 7, 9]
        else:
            self.max_depth_range = max_depth_range
            
        if learning_rate_range is None:
            self.learning_rate_range = [0.01, 0.1, 0.2, 0.3]
        else:
            self.learning_rate_range = learning_rate_range
            
        self.best_n_estimators = None
        self.best_max_depth = None
        self.best_learning_rate = None
        self.best_model = None
        self.model_name = "XGBoost Regression"
        self.parameter_scores = {}
        
    def find_best_parameters(self, X, y, cv=5):
        """
        Find the best hyperparameters using cross-validation
        """
        print(f"Finding best parameters for {self.model_name}...")
        
        best_score = -np.inf
        
        for n_estimators in self.n_estimators_range:
            for max_depth in self.max_depth_range:
                for learning_rate in self.learning_rate_range:
                    # Create and evaluate model
                    model = xgb.XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42,
                        verbosity=0
                    )
                    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                    
                    mean_score = scores.mean()
                    
                    self.parameter_scores[(n_estimators, max_depth, learning_rate)] = {
                        'mean_score': mean_score,
                        'std_score': scores.std(),
                        'scores': scores
                    }
                    
                    print(f"n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}: CV Score = {mean_score:.4f} ± {scores.std():.4f}")
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        self.best_n_estimators = n_estimators
                        self.best_max_depth = max_depth
                        self.best_learning_rate = learning_rate
        
        print(f"\nBest parameters: n_estimators = {self.best_n_estimators}, max_depth = {self.best_max_depth}, learning_rate = {self.best_learning_rate}")
        
        return self.best_n_estimators, self.best_max_depth, self.best_learning_rate
    
    def fit(self, X, y, n_estimators=None, max_depth=None, learning_rate=None):
        """
        Fit the XGBoost model
        """
        if n_estimators is None or max_depth is None or learning_rate is None:
            if self.best_n_estimators is None:
                self.find_best_parameters(X, y)
            n_estimators = self.best_n_estimators
            max_depth = self.best_max_depth
            learning_rate = self.best_learning_rate
        
        # Create and fit the model
        self.best_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=0
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
            'max_depth': self.best_max_depth,
            'learning_rate': self.best_learning_rate
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
            model = xgb.XGBRegressor(
                n_estimators=n_est,
                max_depth=self.best_max_depth,
                learning_rate=self.best_learning_rate,
                random_state=42,
                verbosity=0
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
    
    def plot_trees_analysis(self):
        """
        Analyze individual trees in the ensemble
        """
        if self.best_model is None:
            print("Model not fitted yet. Call fit() first.")
            return
        
        # Get tree information
        booster = self.best_model.get_booster()
        tree_info = booster.get_dump()
        
        n_trees = len(tree_info)
        tree_depths = []
        tree_leaves = []
        
        for tree in tree_info:
            lines = tree.split('\n')
            max_depth = 0
            n_leaves = 0
            
            for line in lines:
                if 'leaf' in line:
                    n_leaves += 1
                elif '[' in line and ']' in line:
                    # Count brackets to estimate depth
                    depth = line.count('[')
                    max_depth = max(max_depth, depth)
            
            tree_depths.append(max_depth)
            tree_leaves.append(n_leaves)
        
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
    
    def plot_parameter_importance(self):
        """
        Plot the importance of different parameters
        """
        if not self.parameter_scores:
            print("No parameter scores available. Run find_best_parameters() first.")
            return
        
        # Analyze each parameter separately
        param_analysis = {}
        
        # n_estimators analysis
        n_est_scores = {}
        for params, score_info in self.parameter_scores.items():
            n_est = params[0]
            if n_est not in n_est_scores:
                n_est_scores[n_est] = []
            n_est_scores[n_est].append(score_info['mean_score'])
        
        for n_est in n_est_scores:
            param_analysis[f'n_estimators_{n_est}'] = np.mean(n_est_scores[n_est])
        
        # max_depth analysis
        depth_scores = {}
        for params, score_info in self.parameter_scores.items():
            depth = params[1]
            if depth not in depth_scores:
                depth_scores[depth] = []
            depth_scores[depth].append(score_info['mean_score'])
        
        for depth in depth_scores:
            param_analysis[f'max_depth_{depth}'] = np.mean(depth_scores[depth])
        
        # learning_rate analysis
        lr_scores = {}
        for params, score_info in self.parameter_scores.items():
            lr = params[2]
            if lr not in lr_scores:
                lr_scores[lr] = []
            lr_scores[lr].append(score_info['mean_score'])
        
        for lr in lr_scores:
            param_analysis[f'learning_rate_{lr}'] = np.mean(lr_scores[lr])
        
        # Plot parameter importance
        plt.figure(figsize=(15, 8))
        
        # Create subplots for each parameter type
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # n_estimators
        n_est_params = {k: v for k, v in param_analysis.items() if k.startswith('n_estimators')}
        n_est_values = [int(k.split('_')[1]) for k in n_est_params.keys()]
        n_est_scores = list(n_est_params.values())
        ax1.bar(range(len(n_est_values)), n_est_scores, color='skyblue')
        ax1.set_xticks(range(len(n_est_values)))
        ax1.set_xticklabels(n_est_values)
        ax1.set_xlabel('Number of Estimators')
        ax1.set_ylabel('Average R² Score')
        ax1.set_title('n_estimators Impact')
        ax1.grid(True, alpha=0.3)
        
        # max_depth
        depth_params = {k: v for k, v in param_analysis.items() if k.startswith('max_depth')}
        depth_values = [int(k.split('_')[1]) for k in depth_params.keys()]
        depth_scores = list(depth_params.values())
        ax2.bar(range(len(depth_values)), depth_scores, color='lightgreen')
        ax2.set_xticks(range(len(depth_values)))
        ax2.set_xticklabels(depth_values)
        ax2.set_xlabel('Max Depth')
        ax2.set_ylabel('Average R² Score')
        ax2.set_title('max_depth Impact')
        ax2.grid(True, alpha=0.3)
        
        # learning_rate
        lr_params = {k: v for k, v in param_analysis.items() if k.startswith('learning_rate')}
        lr_values = [float(k.split('_')[1]) for k in lr_params.keys()]
        lr_scores = list(lr_params.values())
        ax3.bar(range(len(lr_values)), lr_scores, color='lightcoral')
        ax3.set_xticks(range(len(lr_values)))
        ax3.set_xticklabels(lr_values)
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('Average R² Score')
        ax3.set_title('learning_rate Impact')
        ax3.grid(True, alpha=0.3)
        
        # Best parameters highlight
        ax4.text(0.1, 0.8, f'Best Parameters:', fontsize=14, fontweight='bold')
        ax4.text(0.1, 0.7, f'n_estimators: {self.best_n_estimators}', fontsize=12)
        ax4.text(0.1, 0.6, f'max_depth: {self.best_max_depth}', fontsize=12)
        ax4.text(0.1, 0.5, f'learning_rate: {self.best_learning_rate}', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Best Parameters Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_other_models(self, X, y):
        """
        Compare XGBoost with other models
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        # Fit all models
        models = {
            'Linear': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': self.best_model
        }
        
        results = {}
        for name, model in models.items():
            if name == 'XGBoost':
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
        colors = ['skyblue', 'lightgreen', 'gold']
        ax1.bar(names, r2_scores, color=colors)
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.grid(True, alpha=0.3)
        
        # RMSE comparison
        rmse_scores = [results[name]['rmse'] for name in names]
        ax2.bar(names, rmse_scores, color=colors)
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Comparison')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

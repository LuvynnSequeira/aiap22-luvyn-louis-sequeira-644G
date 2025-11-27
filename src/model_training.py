"""
Model Training Module for Phishing Website Detection

Trains multiple models, performs hyperparameter tuning, and selects the best one.   
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles model training, tuning, and selection."""
    
    def __init__(self, random_state=42):
        """Initialize trainer with random seed for reproducibility."""
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def initialize_models(self):
        """Initialize multiple classification models."""
        print("Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_baseline_models(self, X_train, y_train, X_test, y_test):
        """Train baseline models and evaluate with cross-validation."""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        results = {}
        
        if not self.models:
            self.initialize_models()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='f1', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Test predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
            
            print(f"  CV F1-Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test F1-Score: {f1:.4f}")
            print(f"  Test ROC-AUC: {roc_auc:.4f}")
            
            # Track best model
            if f1 > self.best_score:
                self.best_score = f1
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*60)
        print(f"Best baseline model: {self.best_model_name} (F1-Score: {self.best_score:.4f})")
        print("="*60)
        
        return results
    
    def tune_random_forest(self, X_train, y_train):
        """Tune Random Forest hyperparameters using GridSearchCV."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING - RANDOM FOREST")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        print("Performing Grid Search...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            rf, 
            param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_gradient_boosting(self, X_train, y_train):
        """Tune Gradient Boosting hyperparameters using GridSearchCV."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING - GRADIENT BOOSTING")
        print("="*60)
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        
        gb = GradientBoostingClassifier(random_state=self.random_state)
        
        print("Performing Grid Search...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            gb, 
            param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_final_model(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True):
        """Train baseline models, optionally tune top performers, and return best model."""
        # First, train baseline models to identify best approach
        baseline_results = self.train_baseline_models(X_train, y_train, X_test, y_test)
        
        if tune_hyperparameters:
            # Tune the top performing models
            print("\n" + "="*60)
            print("HYPERPARAMETER TUNING FOR TOP MODELS")
            print("="*60)
            
            # Tune Random Forest
            tuned_rf = self.tune_random_forest(X_train, y_train)
            
            # Tune Gradient Boosting
            tuned_gb = self.tune_gradient_boosting(X_train, y_train)
            
            # Evaluate tuned models
            from sklearn.metrics import f1_score
            
            rf_f1 = f1_score(y_test, tuned_rf.predict(X_test))
            gb_f1 = f1_score(y_test, tuned_gb.predict(X_test))
            
            print("\n" + "="*60)
            print("TUNED MODELS PERFORMANCE")
            print("="*60)
            print(f"Tuned Random Forest F1-Score: {rf_f1:.4f}")
            print(f"Tuned Gradient Boosting F1-Score: {gb_f1:.4f}")
            
            # Select best tuned model
            if rf_f1 > gb_f1:
                self.best_model = tuned_rf
                self.best_model_name = "Tuned Random Forest"
                self.best_score = rf_f1
            else:
                self.best_model = tuned_gb
                self.best_model_name = "Tuned Gradient Boosting"
                self.best_score = gb_f1
        
        print("\n" + "="*60)
        print(f"FINAL MODEL SELECTED: {self.best_model_name}")
        print(f"F1-Score: {self.best_score:.4f}")
        print("="*60)
        
        return self.best_model
    
    def save_model(self, filepath='models/phishing_detector.pkl'):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_dict = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='models/phishing_detector.pkl'):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.best_model = model_dict['model']
        self.best_model_name = model_dict['model_name']
        self.best_score = model_dict['score']
        
        print(f"Model loaded from {filepath}")
        print(f"Model: {self.best_model_name}")
        print(f"F1-Score: {self.best_score:.4f}")
        
        return self.best_model


if __name__ == "__main__":
    print("Model Training Module")
    print("This module should be imported and used by the main pipeline script.")


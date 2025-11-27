"""
Model Evaluation Module for Phishing Website Detection

Evaluates model performance and generates visualizations and reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelEvaluator:
    """Handles model evaluation and reporting."""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model and return metrics."""
        print("\n" + "="*60)
        print(f"EVALUATING MODEL: {model_name}")
        print("="*60)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC-AUC if probability predictions available
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = metrics
        
        # Print metrics
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Legitimate (0)', 'Phishing (1)']))
        
        return metrics
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name="Model", save_path=None):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name="Model", save_path=None):
        """Plot and save ROC curve."""
        if y_pred_proba is None:
            print("Cannot plot ROC curve: probability predictions not available")
            return
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name="Model", 
                                top_n=20, save_path=None):
        """Plot feature importance for tree-based models."""
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return importance_df
    
    def generate_evaluation_report(self, model, X_test, y_test, feature_names=None,
                                   model_name="Model", output_dir="results"):
        """Generate comprehensive evaluation report with visualizations."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test, model_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        self.plot_confusion_matrix(
            y_test, 
            metrics['y_pred'], 
            model_name,
            save_path=f"{output_dir}/confusion_matrix.png"
        )
        
        # Plot ROC curve
        if metrics['y_pred_proba'] is not None:
            print("Generating ROC curve...")
            self.plot_roc_curve(
                y_test,
                metrics['y_pred_proba'],
                model_name,
                save_path=f"{output_dir}/roc_curve.png"
            )
        
        # Plot feature importance
        if feature_names and hasattr(model, 'feature_importances_'):
            print("Generating feature importance plot...")
            importance_df = self.plot_feature_importance(
                model,
                feature_names,
                model_name,
                save_path=f"{output_dir}/feature_importance.png"
            )
            
            # Save feature importance to CSV
            importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
            print(f"Feature importance saved to {output_dir}/feature_importance.csv")
        
        # Save metrics to file
        metrics_file = f"{output_dir}/evaluation_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"MODEL EVALUATION REPORT: {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
            if metrics['roc_auc']:
                f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']) + "\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, metrics['y_pred'],
                                         target_names=['Legitimate (0)', 'Phishing (1)']))
        
        print(f"\nEvaluation metrics saved to {metrics_file}")
        
        print("\n" + "="*60)
        print("EVALUATION REPORT GENERATED SUCCESSFULLY")
        print(f"Results saved in: {output_dir}/")
        print("="*60)
        
        return metrics


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("This module should be imported and used by the main pipeline script.")


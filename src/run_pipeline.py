"""
Main Pipeline Script for Phishing Website Detection

Orchestrates the complete ML workflow from data loading to evaluation.
"""

import sys
import os
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def run_full_pipeline(tune_hyperparameters=True, save_models=True):
    """Execute the complete ML pipeline."""
    start_time = datetime.now()
    
    print_header("PHISHING WEBSITE DETECTION - ML PIPELINE")
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hyperparameter tuning: {'Enabled' if tune_hyperparameters else 'Disabled'}")
    print(f"Save models: {'Enabled' if save_models else 'Disabled'}")
    
    try:
        # ========================================
        # STEP 1: DATA PREPROCESSING
        # ========================================
        print_header("STEP 1: DATA PREPROCESSING")
        
        preprocessor = DataPreprocessor(
            db_path='data/phishing.db',
            test_size=0.2,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = preprocessor.run_preprocessing()
        
        if save_models:
            preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        print("\n[SUCCESS] Data preprocessing completed successfully!")
        
        # ========================================
        # STEP 2: MODEL TRAINING
        # ========================================
        print_header("STEP 2: MODEL TRAINING")
        
        trainer = ModelTrainer(random_state=42)
        
        final_model = trainer.train_final_model(
            X_train, y_train, X_test, y_test,
            tune_hyperparameters=tune_hyperparameters
        )
        
        if save_models:
            trainer.save_model('models/phishing_detector.pkl')
        
        print("\n[SUCCESS] Model training completed successfully!")
        
        # ========================================
        # STEP 3: MODEL EVALUATION
        # ========================================
        print_header("STEP 3: MODEL EVALUATION")
        
        evaluator = ModelEvaluator()
        
        metrics = evaluator.generate_evaluation_report(
            model=final_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=preprocessor.feature_names,
            model_name=trainer.best_model_name,
            output_dir='results'
        )
        
        print("\n[SUCCESS] Model evaluation completed successfully!")
        
        # ========================================
        # PIPELINE SUMMARY
        # ========================================
        print_header("PIPELINE EXECUTION SUMMARY")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"Pipeline completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {duration}")
        print(f"\nFinal Model: {trainer.best_model_name}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nModel saved to: models/phishing_detector.pkl")
        print(f"Preprocessor saved to: models/preprocessor.pkl")
        print(f"Results saved to: results/")
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY!")
        
        return {
            'model': final_model,
            'metrics': metrics,
            'preprocessor': preprocessor,
            'trainer': trainer,
            'evaluator': evaluator
        }
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: Pipeline execution failed!")
        print("="*70)
        print(f"\nError message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Phishing Website Detection ML Pipeline'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Disable hyperparameter tuning (faster but may reduce performance)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save models and preprocessor'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_full_pipeline(
        tune_hyperparameters=not args.no_tune,
        save_models=not args.no_save
    )


if __name__ == "__main__":
    main()

